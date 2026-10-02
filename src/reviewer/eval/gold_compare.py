"""Gold vs extracted: closest-claim alignment and number recall. Pure functions, no LLM.

Number rule (also stated in the report header, see ``NUMBER_RULE``):

1. Gold numbers are the tokens of the gold ``result`` field that ``checks.find_numbers`` counts
   as numbers.
2. Each token is reduced to its numeric core(s): leading sign/approx characters
   (``− - + ± ~ ≈ < > ≤ ≥ $``) are dropped, then every run of digits with internal ``.`` or
   ``,`` (``\\d+([.,]\\d+)*``) is a core. So ``48.6%`` → ``48.6``, ``−0.016`` → ``0.016``,
   ``3x``/``2.5×`` → ``3``/``2.5``, ``175B`` → ``175``, a range ``3–5%`` → ``3`` and ``5``.
3. A core is found when it occurs as plain text, with numeric boundaries, in the concatenation
   of every field of every extracted claim of the paper (statement, method, problem, setting,
   result, novelty_assertion, evidence) plus the one-line summary. Numeric boundary: not
   preceded by a digit or by a digit followed by ``.``/``,``, and not followed by a digit or by
   ``.``/``,`` plus a digit. So ``0.5`` does not match inside ``0.503`` or ``10.5``; ``28.4``
   matches in ``28.4,`` and in ``(28.4%)``.
4. A gold number counts as found when all of its cores are found. Number recall per paper =
   found / total over all gold claims of the paper.
"""

import re
from collections.abc import Sequence
from typing import Literal

from reviewer.models import GoldClaims, PaperClaims

from .checks import find_numbers, tfidf_cosine
from .schemas import GoldAlignment, GoldComparison, NumberCheck

NUMBER_RULE = (
    "Gold numbers are the tokens of each gold claim's `result` that count as numbers (same "
    "heuristic as `missing_numbers`). Each is reduced to its numeric core: leading sign "
    "characters (− - + ± ~ ≈) are dropped, and every run of digits with internal `.`/`,` is a "
    "core (`48.6%`→`48.6`, `−0.016`→`0.016`, `3x`→`3`, `175B`→`175`, a range `3–5` → `3` and "
    "`5`). A number is found (✓) when every core appears as plain text in the concatenation "
    "of all fields of all extracted claims of the paper (statement, method, problem, setting, "
    "result, novelty_assertion, evidence) plus the one-line summary, with numeric boundaries: "
    "`0.5` does not match inside `0.503` or `10.5`. Formatting differences count as misses "
    "(`1,000` vs `1000`). No LLM is involved. Number recall = found / total gold numbers."
)

_LEADING_SIGNS = "−-+±~≈<>≤≥$"
_CORE_RE = re.compile(r"\d+(?:[.,]\d+)*")


def number_cores(token: str) -> list[str]:
    """Numeric cores of a number token (see module docstring, rule 2)."""
    return _CORE_RE.findall(token.lstrip(_LEADING_SIGNS))


def contains_number(core: str, haystack: str) -> bool:
    """Plain string match of `core` in `haystack` with numeric boundaries."""
    pattern = rf"(?<!\d)(?<!\d[.,]){re.escape(core)}(?!\d)(?![.,]\d)"
    return re.search(pattern, haystack) is not None


def claims_haystack(claims: PaperClaims) -> str:
    """Every field of every extracted claim plus the one-line summary, newline-joined."""
    parts = [claims.one_line_summary]
    for c in claims.claims:
        parts += [
            c.statement,
            c.method,
            c.problem,
            c.setting,
            c.result,
            c.novelty_assertion,
            *c.evidence,
        ]
    return "\n".join(parts)


def check_numbers(
    result: str | None, haystack: str, field: Literal["result", "setting"] = "result"
) -> list[NumberCheck]:
    checks: list[NumberCheck] = []
    for token in find_numbers(result or ""):
        cores = number_cores(token)
        if not cores:
            continue
        checks.append(
            NumberCheck(
                number=token,
                cores=cores,
                found=all(contains_number(c, haystack) for c in cores),
                field=field,
            )
        )
    return checks


def closest_claim(
    statement: str, candidates: Sequence[str], idf: dict[str, float]
) -> tuple[int | None, float]:
    """(index, cosine) of the candidate most similar to `statement`; ties go to the lowest."""
    best: tuple[int | None, float] = (None, 0.0)
    for j, cand in enumerate(candidates):
        cos = tfidf_cosine(statement, cand, idf)
        if best[0] is None or cos > best[1]:
            best = (j, cos)
    return best


def compare_gold(
    gold: GoldClaims, claims: PaperClaims | None, idf: dict[str, float]
) -> GoldComparison:
    """Align every gold claim with its closest extracted claim and check its result numbers."""
    statements = [c.statement for c in claims.claims] if claims is not None else []
    haystack = claims_haystack(claims) if claims is not None else ""
    alignments: list[GoldAlignment] = []
    for i, g in enumerate(gold.claims):
        idx, score = closest_claim(g.statement, statements, idf)
        alignments.append(
            GoldAlignment(
                gold_index=i,
                closest_index=idx,
                score=score,
                numbers=check_numbers(g.result, haystack)
                + check_numbers(g.setting, haystack, field="setting"),
            )
        )
    return GoldComparison(alignments=alignments)
