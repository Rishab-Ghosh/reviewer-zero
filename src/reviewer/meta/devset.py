"""The 60-paper ICLR dev set (plan §10): a deterministic stratified sample of real OpenReview
submissions over reviewer-index's decisions.jsonl (submission id, venue, decision, and every
reviewer's rating). Pure; no network, no LLM."""

import logging
from pathlib import Path

from reviewer.cache import content_key
from reviewer.models import IclrDecisionRow

logger = logging.getLogger(__name__)

ELIGIBLE_STATUSES = ("accepted", "rejected", "withdrawn")


def load_decisions(path: Path) -> list[IclrDecisionRow]:
    rows = [
        IclrDecisionRow.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    logger.info("loaded %d decision rows from %s", len(rows), path)
    return rows


def mean_rating(row: IclrDecisionRow) -> float | None:
    return sum(row.ratings) / len(row.ratings) if row.ratings else None


def eligible(rows: list[IclrDecisionRow], venues: tuple[str, ...]) -> list[IclrDecisionRow]:
    """Rows in one of `venues`, with a decision status we sample from and at least one rating
    (withdrawn-before-any-review submissions carry no signal for correlation/AUC)."""
    return [
        r for r in rows if r.venue in venues and r.final_status in ELIGIBLE_STATUSES and r.ratings
    ]


def stratified_sample(
    rows: list[IclrDecisionRow],
    cells: list[tuple[str, str]],
    per_cell: int,
    seed_tag: str,
    buffer: int = 5,
) -> dict[tuple[str, str], list[IclrDecisionRow]]:
    """`per_cell` + `buffer` candidates per (venue, final_status) cell, in a deterministic
    hash-sorted order (the buffer lets the fetch step skip PDFs that turn out to be unreachable
    without breaking determinism: always take the next candidate in this fixed order)."""
    by_cell: dict[tuple[str, str], list[IclrDecisionRow]] = {c: [] for c in cells}
    for r in rows:
        key = (r.venue, r.final_status)
        if key in by_cell:
            by_cell[key].append(r)
    out = {}
    for cell, candidates in by_cell.items():
        ordered = sorted(
            candidates, key=lambda r: content_key(seed_tag, cell[0], cell[1], r.submission_id)
        )
        n = per_cell + buffer
        if len(ordered) < n:
            logger.warning(
                "cell %s: only %d eligible candidates for %d wanted (per_cell=%d + buffer=%d)",
                cell,
                len(ordered),
                n,
                per_cell,
                buffer,
            )
        out[cell] = ordered[:n]
    return out
