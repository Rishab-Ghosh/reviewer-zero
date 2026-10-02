"""reports/private/grades.yaml: template, merge (never clobbers human input), load, and summary."""

import logging
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from statistics import mean

import yaml
from pydantic import ValidationError

from reviewer.models import ClaimGrade, ContributionType

from .loader import TIER_ORDER
from .schemas import GradesFile, PaperRecord, ScoreRow, SummaryTable

logger = logging.getLogger(__name__)

_HUMAN_FIELDS = ("specificity", "correctness", "matches_gold", "notes")


# --------------------------------------------------------------------------- template + merge


def build_grades_template(records: Iterable[PaperRecord]) -> list[ClaimGrade]:
    """One blank grade per extracted claim, in report order."""
    return [
        ClaimGrade(
            paper_id=r.entry.paper_id,
            tier=r.entry.tier,
            claim_index=i,
            contribution_type=claim.contribution_type,
            statement=claim.statement,
        )
        for r in records
        if r.claims is not None
        for i, claim in enumerate(r.claims.claims)
    ]


def is_filled(g: ClaimGrade) -> bool:
    return (
        g.specificity is not None
        or g.correctness is not None
        or g.matches_gold is not None
        or bool(g.notes.strip())
    )


def is_graded(g: ClaimGrade) -> bool:
    return g.specificity is not None or g.correctness is not None


def merge_grades(template: Sequence[ClaimGrade], existing: GradesFile) -> GradesFile:
    """Carry human-entered values from `existing` into `template`. Pure.

    An old entry is reused when (paper_id, claim_index, statement) match; failing that, when
    (paper_id, statement) match (the claim moved position). Filled old entries that match
    nothing move to `stale`; blank ones are dropped. Existing stale entries are kept.
    """
    unused = list(existing.grades)

    def take(pred: Callable[[ClaimGrade], bool]) -> ClaimGrade | None:
        for i, g in enumerate(unused):
            if pred(g):
                return unused.pop(i)
        return None

    merged: list[ClaimGrade] = []
    for t in template:
        old = take(
            lambda g, t=t: (
                (g.paper_id, g.claim_index, g.statement) == (t.paper_id, t.claim_index, t.statement)
            )
        ) or take(lambda g, t=t: (g.paper_id, g.statement) == (t.paper_id, t.statement))
        if old is None:
            merged.append(t)
        else:
            merged.append(t.model_copy(update={f: getattr(old, f) for f in _HUMAN_FIELDS}))

    stale = list(existing.stale)
    stale_keys = {_key(g) for g in stale}
    for g in unused:
        if is_filled(g) and _key(g) not in stale_keys:
            stale.append(g)
            stale_keys.add(_key(g))
    return GradesFile(grades=merged, stale=stale)


def _key(g: ClaimGrade) -> tuple[str, int, str]:
    return g.paper_id, g.claim_index, g.statement


# --------------------------------------------------------------------------- io


def _header(gold_paper_ids: Sequence[str]) -> str:
    gold = ", ".join(gold_paper_ids) if gold_paper_ids else "(none yet)"
    return f"""\
# reports/private/grades.yaml — hand grades for the claims in reports/private/week1_claims.md.
# Gitignored: it includes own papers. Do not copy it into a committed file.
#
# One entry per extracted claim, in report order. Fill in:
#   specificity   1-5  could a novelty search built on this claim work? (5 = method, setting,
#                      number; falsifiable and searchable. 1 = could describe 100 papers)
#   correctness   1-5  does the paper make this claim as scoped? (5 = faithful, 1 = wrong)
#   matches_gold  0-based index of the gold claim this claim matches, -1 for no match,
#                 null if not graded yet.
#                 Only meaningful for papers with a gold file: {gold}
#   notes         free text
# Leave a field null / '' if you have not graded it. Do not edit paper_id, tier, claim_index,
# contribution_type or statement: they identify the claim. claim_index is 0-based
# ("Claim 1" in the report is claim_index 0).
#
# Re-running `reviewer eval` keeps your grades. If a claim's statement changes, it gets a fresh
# blank entry and your old grade moves to `stale:` below (it is never deleted).
# Then run `reviewer eval summarize`.
"""


def _dump(grades: Iterable[ClaimGrade]) -> list[dict[str, object]]:
    return [g.model_dump(mode="json") for g in grades]


def load_grades_file(path: Path) -> GradesFile:
    """Parse grades.yaml. Raises ValueError with context if a human-entered value is invalid."""
    if not path.exists():
        return GradesFile()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if isinstance(data, list):  # tolerate a bare list
        data = {"grades": data}
    try:
        return GradesFile.model_validate(
            {"grades": data.get("grades") or [], "stale": data.get("stale") or []}
        )
    except ValidationError as e:
        raise ValueError(f"{path} has invalid entries:\n{e}") from e


def load_grades(path: Path) -> list[ClaimGrade]:
    return load_grades_file(path).grades


def write_grades(
    path: Path, grades: Sequence[ClaimGrade], gold_paper_ids: Sequence[str] = ()
) -> GradesFile:
    """Write `grades` (normally a blank template), merged with any existing file at `path`."""
    existing = load_grades_file(path)
    merged = merge_grades(grades, existing)
    body: dict[str, object] = {"grades": _dump(merged.grades)}
    if merged.stale:
        body["stale"] = _dump(merged.stale)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        _header(gold_paper_ids)
        + "\n"
        + yaml.safe_dump(body, sort_keys=False, allow_unicode=True, width=100)
    )
    path.write_text(text, encoding="utf-8")
    n_kept = sum(is_filled(g) for g in merged.grades)
    logger.info(
        "wrote %s: %d claims (%d with grades kept), %d stale",
        path,
        len(merged.grades),
        n_kept,
        len(merged.stale),
    )
    return merged


# --------------------------------------------------------------------------- summary


def _row(group: str, grades: Sequence[ClaimGrade]) -> ScoreRow:
    spec = [g.specificity for g in grades if g.specificity is not None]
    corr = [g.correctness for g in grades if g.correctness is not None]
    return ScoreRow(
        group=group,
        n_claims=len(grades),
        n_graded=sum(is_graded(g) for g in grades),
        mean_specificity=mean(spec) if spec else None,
        mean_correctness=mean(corr) if corr else None,
    )


def summarize(grades: Sequence[ClaimGrade], gold_paper_ids: Iterable[str] = ()) -> SummaryTable:
    """Mean scores (blanks ignored) by contribution type, by tier, and overall.

    Gold match rate = claims in gold papers with `matches_gold` >= 0 / claims in gold papers
    with `matches_gold` filled in (-1 = no match; null = not graded, excluded).
    """
    gold_ids = set(gold_paper_ids)
    by_type = [
        _row(ct.value, [g for g in grades if g.contribution_type == ct])
        for ct in ContributionType
        if any(g.contribution_type == ct for g in grades)
    ]
    by_tier = [
        _row(tier.value, [g for g in grades if g.tier == tier])
        for tier in TIER_ORDER
        if any(g.tier == tier for g in grades)
    ]
    gold_graded = [g for g in grades if g.paper_id in gold_ids and g.matches_gold is not None]
    gold_matched = sum(g.matches_gold is not None and g.matches_gold >= 0 for g in gold_graded)
    return SummaryTable(
        by_type=by_type,
        by_tier=by_tier,
        overall=_row("overall", grades),
        n_papers=len({g.paper_id for g in grades}),
        n_total=len(grades),
        n_graded=sum(is_graded(g) for g in grades),
        gold_graded=len(gold_graded),
        gold_matched=gold_matched,
        gold_match_rate=gold_matched / len(gold_graded) if gold_graded else None,
    )


def _fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:.2f}"


def _table(title: str, rows: Sequence[ScoreRow]) -> list[str]:
    width = max([len(title), *(len(r.group) for r in rows)])
    out = [
        f"{title:<{width}}  {'graded':>9}  {'specificity':>11}  {'correctness':>11}",
        "-" * (width + 38),
    ]
    for r in rows:
        out.append(
            f"{r.group:<{width}}  {f'{r.n_graded}/{r.n_claims}':>9}  "
            f"{_fmt(r.mean_specificity):>11}  {_fmt(r.mean_correctness):>11}"
        )
    return out


def render_summary(table: SummaryTable) -> str:
    lines = [
        f"Graded {table.n_graded}/{table.n_total} claims across {table.n_papers} papers "
        "(means ignore blanks).",
        "",
        *_table("contribution_type", table.by_type),
        "",
        *_table("tier", table.by_tier),
        "",
        *_table("all", [table.overall]),
        "",
    ]
    if table.gold_match_rate is None:
        lines.append("Gold match rate: - (no graded claims in papers with a gold file)")
    else:
        lines.append(
            f"Gold match rate: {table.gold_match_rate:.0%} "
            f"({table.gold_matched}/{table.gold_graded} graded claims in gold papers)"
        )
    return "\n".join(lines)
