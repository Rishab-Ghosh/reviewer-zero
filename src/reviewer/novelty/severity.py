"""Novelty severity: one LLM call per claim reads the claim and its final top 10 (labels and
reasons) and scores 0-3; the verdict is derived from the score."""

import logging

from reviewer.claims.prompts import Prompt
from reviewer.llm.client import LLMClient, LLMMalformedOutputError
from reviewer.models import (
    JudgedCandidate,
    NoveltyQuery,
    NoveltyVerdict,
    SeverityOutput,
    SeverityResult,
    TokenUsage,
)
from reviewer.novelty.rerank import RerankError, render_claim

logger = logging.getLogger(__name__)

TOP_K = 10
MAX_DRIVERS = 2
SINGLE_DRIVER_PROMPTS = frozenset({"severity_v2"})
"""Prompt versions where a score of 2 or 3 must name exactly one driving work (a single prior
work covering the core contribution), not up to two: v1 allowed 1-2 for any score >= 1 and in
practice used two works to justify 2/3 by adding their components together, which the rule
'many works each sharing one component is still 1' was meant to rule out. Kept separate from v1
so v1 stays reproducible."""


def render_top(ranked: list[JudgedCandidate]) -> str:
    blocks = []
    for j in ranked[:TOP_K]:
        p = j.candidate.paper
        year = p.published_date.year if p.published_date else (p.year or "unknown")
        blocks.append(
            f'<candidate key="{p.key}">\nTitle: {p.title}\nYear: {year}\n'
            f"Label: {j.label.value}\nReason: {j.reason}\n"
            + (f"Listwise reason: {j.listwise_reason}\n" if j.listwise_reason else "")
            + "</candidate>"
        )
    return "<prior_work>\n" + "\n\n".join(blocks) + "\n</prior_work>"


def severity_problems(
    out: SeverityOutput, keys: set[str], single_driver_for_overlap: bool = False
) -> list[str]:
    problems = []
    if not 0 <= out.score <= 3:
        problems.append(f"score {out.score} is not 0-3")
    unknown = [d for d in out.drivers if d not in keys]
    if unknown:
        problems.append(f"drivers {unknown} are not keys of the works shown")
    if out.score >= 1 and not 1 <= len(out.drivers) <= MAX_DRIVERS:
        problems.append(f"score {out.score} needs 1 or 2 driving works, got {len(out.drivers)}")
    elif single_driver_for_overlap and out.score >= 2 and len(out.drivers) != 1:
        problems.append(
            f"score {out.score} needs exactly one driving work (a single prior work must cover "
            f"the core contribution; components spread across papers score at most 1), got "
            f"{len(out.drivers)}: {out.drivers}"
        )
    if out.score == 0 and out.drivers:
        problems.append("score 0 has no driving works")
    return problems


def score_severity(
    client: LLMClient,
    query: NoveltyQuery,
    ranked: list[JudgedCandidate],
    *,
    prompt: Prompt,
    model: str,
    max_retries: int,
) -> tuple[SeverityResult, TokenUsage]:
    if not ranked:
        return SeverityResult(
            score=0,
            drivers=[],
            reason="no candidates were retrieved",
            prompt_version=prompt.version,
            model=model,
        ), TokenUsage()
    keys = {j.candidate.paper.key for j in ranked[:TOP_K]}
    single_driver = prompt.version in SINGLE_DRIVER_PROMPTS
    user = f"{render_claim(query)}\n\n{render_top(ranked)}"
    usage = TokenUsage()
    problem: str | None = None
    for attempt in range(max_retries + 1):
        text = (
            user
            if problem is None
            else (
                f"{user}\n\n<previous_attempt_error>\nAttempt {attempt + 1}: {problem}. Answer "
                "again.\n</previous_attempt_error>"
            )
        )
        try:
            out, call_usage = client.structured(
                system=prompt.text,
                user=text,
                model=model,
                schema=SeverityOutput,
                cache_tag=prompt.version,
            )
        except LLMMalformedOutputError as e:
            usage += e.usage or TokenUsage()
            problem = str(e)
            continue
        usage += call_usage
        problem = "; ".join(severity_problems(out, keys, single_driver)) or None
        if problem is None:
            return SeverityResult(
                score=out.score,
                drivers=out.drivers,
                reason=out.reason,
                prompt_version=prompt.version,
                model=model,
            ), usage
    raise RerankError(f"severity still invalid after {max_retries} retries: {problem}", usage)


def severity_verdict(score: int) -> NoveltyVerdict:
    """0 novel; 1-2 incremental; 3 already done."""
    if score >= 3:
        return NoveltyVerdict.already_done
    if score >= 1:
        return NoveltyVerdict.incremental
    return NoveltyVerdict.novel
