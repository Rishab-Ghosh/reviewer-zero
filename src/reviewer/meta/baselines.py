"""Cheap, checker-free comparisons for the meta-reviewer (plan §4 baselines a/b, §6
contamination probe, §9 build-order step 5: "cheap and should run first").

Baseline (a) rates from title+abstract alone; baseline (b) rates from the full blinded text with
no checker evidence. Both exist to show whether the checkers' findings add anything over a
reviewer who either skimmed or read-but-didn't-verify. The contamination probe asks the model,
from memory alone, whether a given title was accepted at the venue -- a confident correct guess
flags the paper as possibly memorized, and dev/test results are reported split by this flag.

None of these touch calibration or the meta-reviewer's own judgment call; they are separate,
parallel LLM calls over the same paper."""

import logging

from reviewer.claims.prompts import Prompt
from reviewer.llm.client import LLMClient, LLMError, LLMMalformedOutputError
from reviewer.meta.judge import rating_scale
from reviewer.models import BaselineRating, ContaminationProbe, ParsedPaper, TokenUsage

logger = logging.getLogger(__name__)


class BaselineJudgeError(RuntimeError):
    def __init__(self, message: str, usage: TokenUsage | None = None) -> None:
        super().__init__(message)
        self.usage = usage


def _rating_problems(rating: BaselineRating, scale: list[int]) -> list[str]:
    problems = []
    if rating.raw_rating not in scale:
        problems.append(f"raw_rating {rating.raw_rating} is not one of the venue's scale {scale}")
    if not 1 <= rating.confidence <= 5:
        problems.append(f"confidence {rating.confidence} is not 1-5")
    return problems


def _probe_problems(probe: ContaminationProbe) -> list[str]:
    if not 1 <= probe.confidence <= 5:
        return [f"confidence {probe.confidence} is not 1-5"]
    return []


def _abstract_text(paper: ParsedPaper) -> str:
    return f"<title>\n{paper.title}\n</title>\n\n<abstract>\n{paper.abstract}\n</abstract>"


def judge_baseline_rating(
    client: LLMClient,
    user: str,
    *,
    prompt: Prompt,
    model: str,
    venue: str,
    max_retries: int,
) -> tuple[BaselineRating, TokenUsage]:
    """One rating call, retried on malformed/out-of-scale output. Mirrors judge.judge_paper but
    for the simpler evidence-free schema."""
    scale = rating_scale(venue)
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
                schema=BaselineRating,
                cache_tag=f"{prompt.version}:{venue}",
            )
        except LLMMalformedOutputError as e:
            usage += e.usage or TokenUsage()
            problem = str(e)
            continue
        except LLMError as e:
            usage += e.usage or TokenUsage()
            raise BaselineJudgeError(f"{type(e).__name__}: {e}", usage) from e
        usage += call_usage
        problem = "; ".join(_rating_problems(out, scale)) or None
        if problem is None:
            return out, usage
        logger.warning("baseline rating retry %d/%d: %s", attempt + 1, max_retries, problem)
    raise BaselineJudgeError(
        f"baseline rating still invalid after {max_retries} retries: {problem}", usage
    )


def run_baseline_abstract(
    client: LLMClient,
    paper: ParsedPaper,
    *,
    prompt: Prompt,
    model: str,
    venue: str,
    venue_year: int | None,
    max_retries: int,
) -> tuple[BaselineRating, TokenUsage]:
    scale = rating_scale(venue)
    user = (
        f'<venue name="{venue}" year="{venue_year or "unknown"}">\n'
        f"Rating scale: {scale}\n</venue>\n\n{_abstract_text(paper)}"
    )
    return judge_baseline_rating(
        client, user, prompt=prompt, model=model, venue=venue, max_retries=max_retries
    )


def run_baseline_full(
    client: LLMClient,
    blinded_text: str,
    *,
    prompt: Prompt,
    model: str,
    venue: str,
    venue_year: int | None,
    max_retries: int,
) -> tuple[BaselineRating, TokenUsage]:
    scale = rating_scale(venue)
    user = (
        f'<venue name="{venue}" year="{venue_year or "unknown"}">\n'
        f"Rating scale: {scale}\n</venue>\n\n<full_text>\n{blinded_text}\n</full_text>"
    )
    return judge_baseline_rating(
        client, user, prompt=prompt, model=model, venue=venue, max_retries=max_retries
    )


def run_contamination_probe(
    client: LLMClient,
    title: str,
    *,
    prompt: Prompt,
    model: str,
    venue: str,
    venue_year: int | None,
    max_retries: int,
) -> tuple[ContaminationProbe, TokenUsage]:
    user = f'<title>\n{title}\n</title>\n<venue name="{venue}" year="{venue_year or "unknown"}"/>'
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
                schema=ContaminationProbe,
                cache_tag=f"{prompt.version}:{venue}",
            )
        except LLMMalformedOutputError as e:
            usage += e.usage or TokenUsage()
            problem = str(e)
            continue
        except LLMError as e:
            usage += e.usage or TokenUsage()
            raise BaselineJudgeError(f"{type(e).__name__}: {e}", usage) from e
        usage += call_usage
        problem = "; ".join(_probe_problems(out)) or None
        if problem is None:
            return out, usage
        logger.warning("contamination probe retry %d/%d: %s", attempt + 1, max_retries, problem)
    raise BaselineJudgeError(
        f"contamination probe still invalid after {max_retries} retries: {problem}", usage
    )
