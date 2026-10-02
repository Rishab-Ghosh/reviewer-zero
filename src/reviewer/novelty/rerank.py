"""LLM judgment of candidates against a claim, in batches, plus the ranking and verdict rule."""

import logging
from pathlib import Path

from reviewer.claims.prompts import Prompt
from reviewer.llm.client import LLMClient, LLMMalformedOutputError
from reviewer.models import (
    LABEL_ORDER,
    Candidate,
    JudgedCandidate,
    ListwiseRanking,
    NoveltyLabel,
    NoveltyQuery,
    NoveltyVerdict,
    RerankOutput,
    RerankOutputV2,
    RerankOutputV3,
    TokenUsage,
)

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
ABSTRACT_CHAR_LIMIT = 2000
TOP_N = 5
RERANK_SCHEMAS: dict[str, type[RerankOutput] | type[RerankOutputV2] | type[RerankOutputV3]] = {
    "rerank_v1": RerankOutput,
    "rerank_v2": RerankOutputV2,
    "rerank_v3": RerankOutputV3,
}
"""Label definitions live in the schema docstrings too, so each prompt version has its own."""
OVERLAP_LABELS = frozenset(
    {NoveltyLabel.same, NoveltyLabel.subsumes, NoveltyLabel.close, NoveltyLabel.builds_on}
)
"""Labels ranked by listwise (everything but `different`). `builds_on` is included so genuinely
close foundational work is not buried by raw similarity alone; it is still never evidence
against novelty (the verdict rule and the severity prompt both exempt it explicitly)."""


class RerankError(RuntimeError):
    def __init__(self, message: str, usage: TokenUsage | None = None) -> None:
        super().__init__(message)
        self.usage = usage


def load_rerank_prompt(name: str) -> Prompt:
    path = PROMPTS_DIR / f"{name.removesuffix('.md')}.md"
    if not path.is_file():
        raise FileNotFoundError(f"no prompt file {path}")
    return Prompt(version=path.stem, text=path.read_text(encoding="utf-8").strip())


def render_claim(query: NoveltyQuery) -> str:
    lines = [f"Statement: {query.statement}"]
    for label, value in (
        ("Method", query.method),
        ("Problem", query.problem),
        ("Setting", query.setting),
        ("Result", query.result),
    ):
        if value.strip():
            lines.append(f"{label}: {value}")
    return "<claim>\n" + "\n".join(lines) + "\n</claim>"


def render_candidate(i: int, cand: Candidate) -> str:
    p = cand.paper
    year = p.published_date.year if p.published_date else p.year
    abstract = (p.abstract or "").strip()
    if len(abstract) > ABSTRACT_CHAR_LIMIT:
        abstract = abstract[:ABSTRACT_CHAR_LIMIT].rstrip() + " [...]"
    return (
        f'<candidate id="{i}">\nTitle: {p.title or "[unknown title]"}\n'
        f"Year: {year or 'unknown'}\n"
        f"Abstract: {abstract or '[no abstract available]'}\n</candidate>"
    )


def render_user(query: NoveltyQuery, batch: list[Candidate]) -> str:
    cands = "\n\n".join(render_candidate(i, c) for i, c in enumerate(batch))
    return f"{render_claim(query)}\n\n<candidates>\n{cands}\n</candidates>"


def judgment_problems(output: RerankOutput | RerankOutputV2, n: int) -> list[str]:
    ids = [j.candidate_id for j in output.judgments]
    problems = []
    if sorted(ids) != list(range(n)):
        missing = sorted(set(range(n)) - set(ids))
        extra = sorted({i for i in ids if not 0 <= i < n or ids.count(i) > 1})
        problems.append(
            f"expected exactly one judgment for each candidate id 0..{n - 1}; "
            f"missing {missing}, unknown or repeated {extra}"
        )
    return problems


def _with_feedback(
    user: str, problem: str | None, attempt: int, ask: str = "one judgment per candidate"
) -> str:
    if problem is None:
        return user
    return (
        f"{user}\n\n<previous_attempt_error>\nAttempt {attempt}: your previous answer was "
        f"rejected: {problem}. Answer again with {ask}.\n"
        f"</previous_attempt_error>"
    )


def judge_batch(
    client: LLMClient,
    query: NoveltyQuery,
    batch: list[Candidate],
    *,
    prompt: Prompt,
    model: str,
    max_retries: int,
) -> tuple[list[JudgedCandidate], TokenUsage]:
    user = render_user(query, batch)
    usage = TokenUsage()
    problem: str | None = None
    for attempt in range(max_retries + 1):
        if attempt:
            logger.warning("rerank retry %d/%d: %s", attempt, max_retries, problem)
        try:
            out, call_usage = client.structured(
                system=prompt.text,
                user=_with_feedback(user, problem, attempt + 1),
                model=model,
                schema=RERANK_SCHEMAS.get(prompt.version, RerankOutput),
                cache_tag=prompt.version,
            )
        except LLMMalformedOutputError as e:
            usage += e.usage or TokenUsage()
            problem = str(e)
            continue
        usage += call_usage
        problems = judgment_problems(out, len(batch))
        if not problems:
            by_id = {j.candidate_id: j for j in out.judgments}
            judged = [
                JudgedCandidate(candidate=c, label=by_id[i].label, reason=by_id[i].reason)
                for i, c in enumerate(batch)
            ]
            return judged, usage
        problem = "; ".join(problems)
    raise RerankError(f"rerank still invalid after {max_retries} retries: {problem}", usage)


def judge_all(
    client: LLMClient,
    query: NoveltyQuery,
    pool: list[Candidate],
    *,
    prompt: Prompt,
    model: str,
    batch_size: int,
    max_retries: int,
    workers: int = 1,
) -> tuple[list[JudgedCandidate], TokenUsage]:
    """Every batch judged independently, `workers` at a time; results and usage are combined
    in batch order, so the output does not depend on concurrency."""
    from concurrent.futures import ThreadPoolExecutor

    batches = [pool[start : start + batch_size] for start in range(0, len(pool), batch_size)]
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(batches) or 1))) as ex:
        outputs = list(
            ex.map(
                lambda b: judge_batch(
                    client, query, b, prompt=prompt, model=model, max_retries=max_retries
                ),
                batches,
            )
        )
    judged: list[JudgedCandidate] = []
    usage = TokenUsage()
    for batch_judged, batch_usage in outputs:
        judged += batch_judged
        usage += batch_usage
    return judged, usage


def rank(judged: list[JudgedCandidate]) -> list[JudgedCandidate]:
    """Closest first: by label (same, subsumes, close, different), then similarity."""
    order = {label: i for i, label in enumerate(LABEL_ORDER)}
    return sorted(
        judged,
        key=lambda j: (order[j.label], -j.candidate.similarity, j.candidate.paper.key),
    )


def verdict(top: list[JudgedCandidate]) -> NoveltyVerdict:
    """already_done if any of the top works is `same` or `subsumes`; incremental if any is
    `close`; otherwise novel."""
    labels = {j.label for j in top}
    if labels & {NoveltyLabel.same, NoveltyLabel.subsumes}:
        return NoveltyVerdict.already_done
    if NoveltyLabel.close in labels:
        return NoveltyVerdict.incremental
    return NoveltyVerdict.novel


def listwise_problems(output: ListwiseRanking, n: int) -> list[str]:
    ids = [item.candidate_id for item in output.ranking]
    if sorted(ids) == list(range(n)):
        return []
    missing = sorted(set(range(n)) - set(ids))
    extra = sorted({i for i in ids if not 0 <= i < n or ids.count(i) > 1})
    return [
        f"rank every candidate id 0..{n - 1} exactly once; missing {missing}, "
        f"unknown or repeated {extra}"
    ]


def listwise_rerank(
    client: LLMClient,
    query: NoveltyQuery,
    ranked: list[JudgedCandidate],
    *,
    prompt: Prompt,
    model: str,
    cap: int,
    max_retries: int,
) -> tuple[list[JudgedCandidate], TokenUsage]:
    """Re-order the top `cap` same/subsumes/close candidates with one listwise call; the rest
    keep their order after them. Labels are unchanged (the verdict rule still uses them)."""
    head = [j for j in ranked if j.label in OVERLAP_LABELS][:cap]
    if len(head) < 2:
        return ranked, TokenUsage()
    user = render_user(query, [j.candidate for j in head])
    usage = TokenUsage()
    problem: str | None = None
    for attempt in range(max_retries + 1):
        if attempt:
            logger.warning("listwise retry %d/%d: %s", attempt, max_retries, problem)
        try:
            out, call_usage = client.structured(
                system=prompt.text,
                user=_with_feedback(
                    user, problem, attempt + 1, ask="every candidate ranked exactly once"
                ),
                model=model,
                schema=ListwiseRanking,
                cache_tag=prompt.version,
            )
        except LLMMalformedOutputError as e:
            usage += e.usage or TokenUsage()
            problem = str(e)
            continue
        usage += call_usage
        problem = "; ".join(listwise_problems(out, len(head))) or None
        if problem is None:
            reordered = [
                head[item.candidate_id].model_copy(
                    update={"listwise_rank": pos, "listwise_reason": item.reason}
                )
                for pos, item in enumerate(out.ranking, start=1)
            ]
            in_head = {id(j) for j in head}
            return reordered + [j for j in ranked if id(j) not in in_head], usage
    raise RerankError(
        f"listwise ranking still invalid after {max_retries} retries: {problem}", usage
    )
