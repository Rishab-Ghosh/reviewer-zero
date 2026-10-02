"""Two-run consensus, the product default: check each paper twice (independent replicates) and
sort what was raised by agreement.

- A checklist item is **confirmed** when both runs give `no`, and **possible** when one does.
  An item neither run raises keeps the shared verdict, or `unclear` if the runs disagree.
- An other concern is **confirmed** when the second run has a matching one (a shared verified
  quote, or content-word overlap >= `MATCH_JACCARD`), else **possible**. Other concerns are
  never matched to checklist items, so a concern filed under different headings in the two runs
  lands in "possible": the rule can under-confirm, never over-confirm.

Why: a single run's verdicts flip on ~11% of items between independent runs (STATUS,
stability experiment); a finding both runs raise is the one to show first. Pure except
`check_paper_twice`.
"""

import re

from reviewer.claims.prompts import Prompt
from reviewer.llm.client import LLMClient
from reviewer.methods.checker import check_paper
from reviewer.methods.quotes import squash
from reviewer.models import (
    CheckedConcern,
    CheckedFinding,
    MethodsVerdict,
    PaperMethodsReview,
    ParsedPaper,
    QuoteCheck,
)

CONFIRMED = "confirmed"
POSSIBLE = "possible"
MATCH_JACCARD = 0.2
"""Set on 2026-09-27 by reading every best-matching concern pair on the 10 stability papers:
all pairs at 0.19-0.32 were the same concern reworded (e.g. 'Absolute R2 values (0.06-0.21)
are very low...' in both runs, 0.27), the first different one scored 0.16. At the first
guess, 0.35, 26% of other concerns were confirmed; at 0.2, see STATUS. Chosen on the same
papers it is reported on, so treat the other-concern rate as optimistic until checked on
fresh papers."""
_WORD = re.compile(r"[a-z][a-z0-9\-]{3,}")
_STOP = frozenset(
    (
        "that",
        "this",
        "with",
        "from",
        "their",
        "there",
        "which",
        "while",
        "paper",
        "method",
        "methods",
        "results",
        "result",
        "experiments",
        "experiment",
        "would",
        "could",
        "should",
        "does",
        "only",
        "about",
        "into",
        "than",
        "more",
        "most",
        "other",
        "such",
        "these",
        "those",
        "were",
        "been",
        "being",
        "have",
        "when",
        "whether",
    )
)


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def _verified(checks: list[QuoteCheck]) -> list[str]:
    return [squash(q.quote) for q in checks if q.verified]


def concerns_match(a: CheckedConcern, b: CheckedConcern) -> bool:
    qa, qb = _verified(a.quotes), _verified(b.quotes)
    if any(x in y or y in x for x in qa for y in qb):
        return True
    wa, wb = _words(a.concern), _words(b.concern)
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= MATCH_JACCARD


def _union_quotes(a: list[QuoteCheck], b: list[QuoteCheck]) -> list[QuoteCheck]:
    seen = {squash(q.quote) for q in a}
    return a + [q for q in b if squash(q.quote) not in seen]


def combine(r0: PaperMethodsReview, r1: PaperMethodsReview) -> PaperMethodsReview:
    """One review from two runs of the same checker on the same paper."""
    by1 = {f.item: f for f in r1.findings}
    findings: list[CheckedFinding] = []
    for f0 in r0.findings:
        f1 = by1[f0.item]
        no0, no1 = f0.verdict == MethodsVerdict.no, f1.verdict == MethodsVerdict.no
        if no0 and no1:
            findings.append(
                f0.model_copy(
                    update={"agreement": CONFIRMED, "quotes": _union_quotes(f0.quotes, f1.quotes)}
                )
            )
        elif no0 or no1:
            findings.append((f0 if no0 else f1).model_copy(update={"agreement": POSSIBLE}))
        elif f0.verdict == f1.verdict:
            findings.append(f0)
        else:
            findings.append(
                f0.model_copy(
                    update={
                        "verdict": MethodsVerdict.unclear,
                        "rationale": f"runs disagree ({f0.verdict.value} / {f1.verdict.value}). "
                        + f0.rationale,
                    }
                )
            )
    left = [c for c in r1.other_concerns if c.verified]
    concerns: list[CheckedConcern] = []
    for c0 in (c for c in r0.other_concerns if c.verified):
        match = next((c1 for c1 in left if concerns_match(c0, c1)), None)
        if match is not None:
            left.remove(match)
            concerns.append(
                c0.model_copy(
                    update={
                        "agreement": CONFIRMED,
                        "quotes": _union_quotes(c0.quotes, match.quotes),
                    }
                )
            )
        else:
            concerns.append(c0.model_copy(update={"agreement": POSSIBLE}))
    concerns += [c.model_copy(update={"agreement": POSSIBLE}) for c in left]
    return r0.model_copy(
        update={
            "findings": findings,
            "other_concerns": concerns,
            "token_usage": r0.token_usage + r1.token_usage,
            "runs": 2,
        }
    )


def bucket_counts(r: PaperMethodsReview) -> dict[str, int]:
    return {
        "items_confirmed": sum(f.agreement == CONFIRMED for f in r.findings),
        "items_possible": sum(f.agreement == POSSIBLE for f in r.findings),
        "other_confirmed": sum(c.agreement == CONFIRMED for c in r.other_concerns),
        "other_possible": sum(c.agreement == POSSIBLE for c in r.other_concerns),
    }


def check_paper_twice(
    client: LLMClient,
    paper: ParsedPaper,
    *,
    tier: str,
    prompt: Prompt,
    repair_prompt: Prompt | None,
    model: str,
    effort: str,
    max_retries: int,
) -> PaperMethodsReview:
    """Replicate 0 (the ordinary cache entry) and replicate 1, combined."""
    runs = [
        check_paper(
            client,
            paper,
            tier=tier,
            prompt=prompt,
            repair_prompt=repair_prompt,
            model=model,
            effort=effort,
            max_retries=max_retries,
            replicate=k,
        )
        for k in (0, 1)
    ]
    return combine(*runs)
