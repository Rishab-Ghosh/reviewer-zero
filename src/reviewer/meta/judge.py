"""The meta-reviewer's LLM judgment call: MetaInput in, MetaJudgment out, k times (plan §4.1).
Calibration (turning judgments into a score) is separate, in calibration.py, and never touches
the LLM."""

import logging

from reviewer.claims.prompts import Prompt
from reviewer.llm.client import LLMClient, LLMError, LLMMalformedOutputError
from reviewer.models import CheckerRunStatus, MetaInput, MetaJudgment, TokenUsage

logger = logging.getLogger(__name__)

VENUE_SCALES: dict[str, list[int]] = {
    "ICLR": [1, 3, 5, 6, 8, 10],
    "NeurIPS": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}
DEFAULT_SCALE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


class MetaJudgeError(RuntimeError):
    def __init__(self, message: str, usage: TokenUsage | None = None) -> None:
        super().__init__(message)
        self.usage = usage


def rating_scale(venue: str) -> list[int]:
    return VENUE_SCALES.get(venue, DEFAULT_SCALE)


def render_evidence(mi: MetaInput) -> str:
    if not mi.evidence:
        return "<evidence>\n(none)\n</evidence>"
    blocks = [f'<evidence id="{e.id}">\n{e.text}\n</evidence>' for e in mi.evidence]
    return "<evidence>\n" + "\n\n".join(blocks) + "\n</evidence>"


def render_checker_status(mi: MetaInput) -> str:
    lines = [
        f"- {s.checker.value}: {s.status.value}" + (f" ({s.detail})" if s.detail else "")
        for s in mi.checker_status
    ]
    return "<checker_status>\n" + "\n".join(lines) + "\n</checker_status>"


def render_input(mi: MetaInput) -> str:
    scale = rating_scale(mi.venue)
    return (
        f'<paper venue="{mi.venue}" venue_year="{mi.venue_year or "unknown"}">\n'
        f"Rating scale for this venue: {scale}\n"
        f"</paper>\n\n"
        f"{render_checker_status(mi)}\n\n"
        f"<full_text>\n{mi.blinded_text}\n</full_text>\n\n"
        f"{render_evidence(mi)}"
    )


def judgment_problems(judgment: MetaJudgment, scale: list[int]) -> list[str]:
    problems = []
    if judgment.raw_rating not in scale:
        problems.append(f"raw_rating {judgment.raw_rating} is not one of the venue's scale {scale}")
    if not 1 <= judgment.confidence <= 5:
        problems.append(f"confidence {judgment.confidence} is not 1-5")
    for name, sub in (
        ("soundness", judgment.soundness),
        ("presentation", judgment.presentation),
        ("contribution", judgment.contribution),
    ):
        if not 1 <= sub.score <= 4:
            problems.append(f"{name} score {sub.score} is not 1-4")
    if not judgment.weaknesses:
        problems.append("at least one weakness is required, even for a strong paper")
    return problems


def reconcile_judgment(judgment: MetaJudgment, known_ids: set[str]) -> tuple[MetaJudgment, int]:
    """Drop any strength/weakness whose evidence_ids do not all resolve against `known_ids`;
    return the cleaned judgment and how many items were dropped (plan §3: 'code drops any item
    whose ids don't resolve ... and counts the drops')."""
    dropped = 0
    kept_strengths = []
    for s in judgment.strengths:
        if all(i in known_ids for i in s.evidence_ids):
            kept_strengths.append(s)
        else:
            dropped += 1
            logger.warning("dropping strength with unresolved evidence ids: %s", s.evidence_ids)
    kept_weaknesses = []
    for w in judgment.weaknesses:
        if all(i in known_ids for i in w.evidence_ids):
            kept_weaknesses.append(w)
        else:
            dropped += 1
            logger.warning("dropping weakness with unresolved evidence ids: %s", w.evidence_ids)
    return judgment.model_copy(
        update={"strengths": kept_strengths, "weaknesses": kept_weaknesses}
    ), dropped


def judge_paper(
    client: LLMClient,
    mi: MetaInput,
    *,
    prompt: Prompt,
    model: str,
    max_retries: int,
) -> tuple[MetaJudgment, int, TokenUsage]:
    """One judgment call. Raises MetaJudgeError after retries. Returns (reconciled judgment,
    dropped-item count, usage)."""
    scale = rating_scale(mi.venue)
    user = render_input(mi)
    known_ids = {e.id for e in mi.evidence}
    usage = TokenUsage()
    problem: str | None = None
    for attempt in range(max_retries + 1):
        text = (
            user
            if problem is None
            else (
                f"{user}\n\n<previous_attempt_error>\nAttempt {attempt + 1}: your previous "
                f"answer was rejected: {problem}. Answer again.\n</previous_attempt_error>"
            )
        )
        try:
            out, call_usage = client.structured(
                system=prompt.text,
                user=text,
                model=model,
                schema=MetaJudgment,
                cache_tag=f"{prompt.version}:{mi.venue}",
            )
        except LLMMalformedOutputError as e:
            usage += e.usage or TokenUsage()
            problem = str(e)
            continue
        except LLMError as e:
            usage += e.usage or TokenUsage()
            raise MetaJudgeError(f"{type(e).__name__}: {e}", usage) from e
        usage += call_usage
        problem = "; ".join(judgment_problems(out, scale)) or None
        if problem is None:
            reconciled, dropped = reconcile_judgment(out, known_ids)
            return reconciled, dropped, usage
        logger.warning("meta judgment retry %d/%d: %s", attempt + 1, max_retries, problem)
    raise MetaJudgeError(
        f"meta judgment still invalid after {max_retries} retries: {problem}", usage
    )


def judge_paper_k(
    client: LLMClient,
    mi: MetaInput,
    *,
    prompt: Prompt,
    model: str,
    max_retries: int,
    k: int,
) -> tuple[list[MetaJudgment], int, TokenUsage]:
    """k independent judgment calls (the LLM cache makes repeats of k=1 free; k>1 needs distinct
    cache tags per run, handled by the client's cache key including the run index)."""
    judgments = []
    total_dropped = 0
    usage = TokenUsage()
    for run in range(k):
        run_prompt = (
            prompt
            if k == 1
            else prompt.model_copy(update={"version": f"{prompt.version}+run{run}"})
        )
        j, dropped, u = judge_paper(
            client, mi, prompt=run_prompt, model=model, max_retries=max_retries
        )
        judgments.append(j)
        total_dropped += dropped
        usage += u
    return judgments, total_dropped, usage


def checker_ok_count(mi: MetaInput) -> int:
    return sum(s.status == CheckerRunStatus.ok for s in mi.checker_status)
