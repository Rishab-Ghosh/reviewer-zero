"""Evaluate the methodology checker against human critiques (recall) and an audited sample of
its concerns (precision).

Recall: one matcher call per critiqued paper sees the critiques and the checker's raised
findings (text only). Strict counts only verified `no` findings and verified other concerns;
lenient also counts `unclear` findings and unverified concerns. Precision: a seeded sample of
strict concerns, each audited by a verifier that sees the full paper. Everything except the two
`*_paper` calls is pure.
"""

import logging
import random
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel

from reviewer.claims.prompts import Prompt
from reviewer.llm.client import LLMClient, LLMMalformedOutputError
from reviewer.methods.checker import MethodsError, other_key, paper_block
from reviewer.models import (
    ConcernCheck,
    ConcernCheckOutput,
    CritiqueMatch,
    CritiqueMatchLabel,
    CritiqueMatchOutput,
    Critiques,
    CritiqueScope,
    MethodsVerdict,
    PaperMethodsReview,
    ParsedPaper,
    TokenUsage,
)

logger = logging.getLogger(__name__)

PRECISION_SEED = 20260925
PRECISION_N = 20


@dataclass(frozen=True)
class Raised:
    """One concern the checker raised, as the matcher and verifier see it."""

    paper_id: str
    key: str
    """Checklist item id, or 'other:N'."""
    status: str
    """'no', 'unclear', 'downgraded' (model said no/yes/na, no quote verified), 'other',
    'other_unverified', or for two-run reviews 'possible' / 'other_possible' (one run only;
    not strict)."""
    text: str
    quotes: tuple[str, ...]

    @property
    def strict(self) -> bool:
        return self.status in ("no", "other")

    @property
    def id(self) -> str:
        return f"{self.paper_id}:{self.key}"


def raised_concerns(review: PaperMethodsReview) -> list[Raised]:
    """Every no/unclear finding and every other concern (lenient set); `strict` marks the
    verified ones."""
    out = []
    for f in review.findings:
        if f.verdict not in (MethodsVerdict.no, MethodsVerdict.unclear):
            continue
        status = "downgraded" if f.downgraded else f.verdict.value
        if f.agreement == "possible":
            status = "possible"  # raised by one of two runs: lenient only
        text = f"{f.concern or f.rationale} (Rationale: {f.rationale})"
        quotes = tuple(q.quote for q in f.quotes if q.verified)
        out.append(Raised(review.paper_id, f.item.value, status, text, quotes))
    for i, c in enumerate(review.other_concerns):
        status = "other" if c.verified else "other_unverified"
        if c.agreement == "possible":
            status = "other_possible"
        quotes = tuple(q.quote for q in c.quotes if q.verified)
        out.append(Raised(review.paper_id, other_key(i), status, c.concern, quotes))
    return out


def critique_text(entry_limitation: str, explanation: str | None) -> str:
    return entry_limitation + (f" — {explanation}" if explanation else "")


def match_user_message(critiques: Critiques, raised: list[Raised]) -> str:
    crit = "\n".join(
        f'<critique index="{i}">{critique_text(e.limitation, e.explanation)}</critique>'
        for i, e in enumerate(critiques.entries)
    )
    finds = "\n".join(f'<finding id="{r.key}">{r.text}</finding>' for r in raised)
    return f"<critiques>\n{crit}\n</critiques>\n\n<findings>\n{finds or '(none)'}\n</findings>"


def match_problems(out: CritiqueMatchOutput, n: int, keys: set[str]) -> list[str]:
    problems = []
    ids = sorted(m.critique_index for m in out.matches)
    if ids != list(range(n)):
        problems.append(f"expected one entry per critique index 0..{n - 1}, got {ids}")
    for m in out.matches:
        if bad := [k for k in m.finding_ids if k not in keys]:
            problems.append(f"critique {m.critique_index}: unknown finding ids {bad}")
        if m.label != CritiqueMatchLabel.none and not m.finding_ids:
            problems.append(f"critique {m.critique_index}: {m.label.value} needs finding_ids")
    return problems


def _structured_with_retries[T: BaseModel](
    client: LLMClient,
    *,
    system: str,
    user: str,
    model: str,
    schema: type[T],
    cache_tag: str,
    check: Callable[[T], list[str]],
    max_retries: int,
    label: str,
) -> tuple[T, TokenUsage]:
    usage = TokenUsage()
    problem: str | None = None
    for attempt in range(max_retries + 1):
        text = (
            user
            if problem is None
            else f"{user}\n\n<previous_attempt_error>\nYour previous answer was rejected: "
            f"{problem}. Answer again.\n</previous_attempt_error>"
        )
        try:
            out, u = client.structured(
                system=system, user=text, model=model, schema=schema, cache_tag=cache_tag
            )
        except LLMMalformedOutputError as e:
            usage += e.usage or TokenUsage()
            problem = str(e)
            continue
        usage += u
        problem = "; ".join(check(out)) or None
        if problem is None:
            return out, usage
        logger.warning("%s retry %d: %s", label, attempt + 1, problem)
    raise MethodsError(f"{label} still invalid after {max_retries} retries: {problem}", usage)


def match_paper(
    client: LLMClient,
    critiques: Critiques,
    review: PaperMethodsReview,
    *,
    prompt: Prompt,
    model: str,
    max_retries: int,
) -> tuple[list[CritiqueMatch], TokenUsage]:
    raised = raised_concerns(review)
    keys = {r.key for r in raised}
    out, usage = _structured_with_retries(
        client,
        system=prompt.text,
        user=match_user_message(critiques, raised),
        model=model,
        schema=CritiqueMatchOutput,
        cache_tag=f"{prompt.version}|{review.prompt_version}",
        check=lambda o: match_problems(o, len(critiques.entries), keys),
        max_retries=max_retries,
        label=f"match {critiques.paper_id}",
    )
    return sorted(out.matches, key=lambda m: m.critique_index), usage


# --------------------------------------------------------------------------- recall


@dataclass(frozen=True)
class CritiqueOutcome:
    paper_id: str
    index: int
    text: str
    scope: CritiqueScope
    lenient: CritiqueMatchLabel
    strict: CritiqueMatchLabel
    finding_ids: tuple[str, ...]
    reason: str


def outcomes(
    critiques: Critiques, review: PaperMethodsReview, matches: list[CritiqueMatch]
) -> list[CritiqueOutcome]:
    """Lenient = the matcher's label; strict = that label only if a matched finding is in the
    strict set, else none."""
    strict_keys = {r.key for r in raised_concerns(review) if r.strict}
    out = []
    for m in matches:
        e = critiques.entries[m.critique_index]
        hit = any(k in strict_keys for k in m.finding_ids)
        strict = m.label if hit else CritiqueMatchLabel.none
        out.append(
            CritiqueOutcome(
                paper_id=critiques.paper_id,
                index=m.critique_index,
                text=critique_text(e.limitation, e.explanation),
                scope=m.scope,
                lenient=m.label,
                strict=strict,
                finding_ids=tuple(m.finding_ids),
                reason=m.reason,
            )
        )
    return out


@dataclass(frozen=True)
class Recall:
    n: int
    match: int
    partial: int

    @property
    def full(self) -> float:
        return self.match / self.n if self.n else 0.0

    @property
    def with_partial(self) -> float:
        return (self.match + self.partial) / self.n if self.n else 0.0


def recall(items: list[CritiqueOutcome], mode: str, scope: CritiqueScope | None = None) -> Recall:
    picked = [o for o in items if scope is None or o.scope == scope]
    labels = [getattr(o, mode) for o in picked]
    return Recall(
        n=len(picked),
        match=sum(label == CritiqueMatchLabel.match for label in labels),
        partial=sum(label == CritiqueMatchLabel.partial for label in labels),
    )


# --------------------------------------------------------------------------- precision


def precision_sample(
    reviews: list[PaperMethodsReview], n: int = PRECISION_N, seed: int = PRECISION_SEED
) -> list[Raised]:
    """A seeded sample of strict concerns across all papers, sorted by paper then key."""
    pool = sorted(
        (r for rev in reviews for r in raised_concerns(rev) if r.strict),
        key=lambda r: (r.paper_id, r.key),
    )
    picked = random.Random(seed).sample(pool, min(n, len(pool)))
    return sorted(picked, key=lambda r: (r.paper_id, r.key))


def verify_user_message(paper: ParsedPaper, concerns: list[Raised]) -> str:
    blocks = "\n\n".join(f'<concern id="{r.key}">\n{r.text}\n</concern>' for r in concerns)
    return f"{paper_block(paper)}\n\n<concerns>\n{blocks}\n</concerns>"


def verify_paper(
    client: LLMClient,
    paper: ParsedPaper,
    concerns: list[Raised],
    *,
    prompt: Prompt,
    model: str,
    max_retries: int,
) -> tuple[list[ConcernCheck], TokenUsage]:
    keys = {r.key for r in concerns}

    def check(o: ConcernCheckOutput) -> list[str]:
        got = sorted(c.concern_id for c in o.checks)
        return [] if got == sorted(keys) else [f"expected checks for {sorted(keys)}, got {got}"]

    out, usage = _structured_with_retries(
        client,
        system=prompt.text,
        user=verify_user_message(paper, concerns),
        model=model,
        schema=ConcernCheckOutput,
        cache_tag=prompt.version,
        check=check,
        max_retries=max_retries,
        label=f"verify {paper.paper_id}",
    )
    return out.checks, usage
