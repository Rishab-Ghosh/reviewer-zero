"""critic_v5: per-claim verification with verbatim quotes, checked in code.

For each claim of the critic's final set, the model lists every number and conclusion with a
verbatim quote from the paper. Code checks (pure, below) that every quote appears in the paper
text the model was shown and that every number in the returned claim is covered by an item
whose quote contains it. Failures go back to the model; a claim that still cannot be verified is
dropped, never kept.
"""

import logging
import re
import unicodedata

from reviewer.claims.grounding import (
    _tuple_in_paper,
    is_grounded,
    numbers_in,
    paper_number_forms,
)
from reviewer.claims.prompts import Prompt
from reviewer.llm.client import LLMClient, LLMError, LLMMalformedOutputError, MissingAPIKeyError
from reviewer.models import (
    Claim,
    ClaimVerification,
    TokenUsage,
    VerificationRecord,
)

logger = logging.getLogger(__name__)

CHECKED_FIELDS = ("statement", "setting", "result")


class VerificationError(RuntimeError):
    """Verification could not run (API failure); the paper fails, no claim is dropped."""


ELLIPSIS_RE = re.compile(r"\s*(?:\.\.\.|…)\s*")
MIN_SEGMENT_CHARS = 8
_TRANSLATE = str.maketrans(
    {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "−": "-", " ": " "}
)


def normalize(text: str) -> str:
    """For quote matching: unicode-normalized, typographic quotes/dashes flattened, lowercase,
    whitespace collapsed."""
    text = unicodedata.normalize("NFKC", text).translate(_TRANSLATE).lower()
    return " ".join(text.split())


def quote_in_paper(quote: str, paper_norm: str) -> bool:
    """Every '...'-separated segment of the quote appears in the paper, in order."""
    segments = [normalize(s) for s in ELLIPSIS_RE.split(quote) if s.strip()]
    if not segments or sum(len(s) for s in segments) < MIN_SEGMENT_CHARS:
        return False
    pos = 0
    for seg in segments:
        found = paper_norm.find(seg, pos)
        if found < 0:
            return False
        pos = found + len(seg)
    return True


def verification_problems(out: ClaimVerification, paper_norm: str) -> list[str]:
    """Why the model's answer cannot be accepted as verified (empty = verified)."""
    if out.verdict == "drop":
        return []
    if out.claim is None:
        return [f"verdict {out.verdict} needs the claim to keep"]
    problems = []
    for it in out.items:
        if not quote_in_paper(it.quote, paper_norm):
            problems.append(
                f"quote for {it.element!r} ({it.field}) is not verbatim in the paper: "
                f"{it.quote[:120]!r}"
            )
    verified = [it for it in out.items if quote_in_paper(it.quote, paper_norm)]
    for field in CHECKED_FIELDS:
        for written, value, d in numbers_in(getattr(out.claim, field)):
            covered = False
            for it in verified:
                if it.field != field and written not in it.element:
                    continue
                if is_grounded(value, d, paper_number_forms(it.quote)) or _tuple_in_paper(
                    written, it.quote
                ):
                    covered = True
                    break
            if not covered:
                problems.append(
                    f"number {written!r} in the {field} has no verified quote containing it"
                )
    return problems


def _feedback(user: str, problems: list[str], attempt: int) -> str:
    return (
        f"{user}\n\n<verification_errors>\nAttempt {attempt}: a code check rejected your "
        "answer:\n- " + "\n- ".join(problems) + "\nFix every quote so it is copied verbatim "
        "from the paper and contains the number it supports, correct or remove unsupported "
        "elements, or drop the claim.\n</verification_errors>"
    )


def verify_claim(
    client: LLMClient,
    claim: Claim,
    index: int,
    *,
    paper_context: str,
    prompt: Prompt,
    model: str,
    max_retries: int,
) -> tuple[Claim | None, VerificationRecord, TokenUsage]:
    """(claim to keep or None, record, usage). Unverifiable after retries -> dropped."""
    paper_norm = normalize(paper_context)
    user = f"{paper_context}\n\n<claim>\n{claim.model_dump_json(indent=2)}\n</claim>"
    usage = TokenUsage()
    problems: list[str] = []
    attempts = 0
    for attempt in range(max_retries + 1):
        attempts = attempt + 1
        text = user if not problems else _feedback(user, problems, attempt + 1)
        try:
            out, call_usage = client.structured(
                system=prompt.text,
                user=text,
                model=model,
                schema=ClaimVerification,
                cache_tag=prompt.version,
            )
        except LLMMalformedOutputError as e:
            usage += e.usage or TokenUsage()
            problems = [str(e)]
            continue
        except MissingAPIKeyError:
            raise
        except LLMError as e:
            # An API failure (network, 5xx, refusal, truncation) says nothing about the claim:
            # fail the paper instead of dropping a claim that was never checked.
            raise VerificationError(f"claim {index}: {type(e).__name__}: {e}") from e
        usage += call_usage
        if out.verdict == "drop":
            rec = VerificationRecord(
                input_index=index, outcome="dropped", reason=out.reason, attempts=attempts
            )
            return None, rec, usage
        problems = verification_problems(out, paper_norm)
        if not problems:
            assert out.claim is not None
            outcome = "kept" if out.claim == claim else "fixed"
            rec = VerificationRecord(
                input_index=index,
                outcome=outcome,
                reason=out.reason,
                items=out.items,
                attempts=attempts,
            )
            return out.claim, rec, usage
    logger.warning("claim %d dropped as unverifiable: %s", index, problems)
    rec = VerificationRecord(
        input_index=index,
        outcome="dropped",
        reason="could not be verified: every number and conclusion needs a verbatim quote",
        attempts=attempts,
        problems=problems,
    )
    return None, rec, usage


def verify_claims(
    client: LLMClient,
    claims: list[Claim],
    *,
    paper_context: str,
    prompt: Prompt,
    model: str,
    max_retries: int,
) -> tuple[list[Claim], list[VerificationRecord], TokenUsage]:
    kept: list[Claim] = []
    records: list[VerificationRecord] = []
    usage = TokenUsage()
    for i, c in enumerate(claims):
        out, rec, u = verify_claim(
            client,
            c,
            i,
            paper_context=paper_context,
            prompt=prompt,
            model=model,
            max_retries=max_retries,
        )
        usage += u
        records.append(rec)
        if out is not None:
            kept.append(out)
    return kept, records, usage
