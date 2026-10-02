"""OpenReview submission -> ParsedPaper for the 60-paper dev run (plan §10). Reuses the methods
review-eval's `OpenReviewParser` (PDF fetch + GROBID parse, disk-cached) so this never duplicates
that logic; adds only the copy into data/parsed/ that the meta batch reads from, and the
deterministic buffer-consuming fetch loop over `devset.stratified_sample`'s candidate cells."""

import logging

from reviewer.config import Settings
from reviewer.methods.review_commands import OpenReviewParser
from reviewer.methods.review_eval import Submission
from reviewer.models import IclrDecisionRow, ParsedPaper
from reviewer.novelty.evaluate import venue_cutoff

logger = logging.getLogger(__name__)


def to_submission(row: IclrDecisionRow) -> Submission:
    return Submission(
        submission_id=row.submission_id,
        venue=row.venue,
        title=row.title,
        cutoff_date=venue_cutoff(row.venue),
        final_status=row.final_status,
        concerns=(),
    )


def register_parsed(settings: Settings, paper: ParsedPaper) -> None:
    """Copy into data/parsed/ (what `meta.commands.run_batch` reads from) -- separate from
    data/openreview/parsed/, which is `OpenReviewParser`'s own cache keyed by `or-<id>` too, so
    this is a plain copy, never a re-parse."""
    out = settings.parsed_dir / f"{paper.paper_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(paper.model_dump_json(indent=2), encoding="utf-8")


def fetch_dev_set(
    candidates: dict[tuple[str, str], list[IclrDecisionRow]],
    target_per_cell: int,
    settings: Settings,
    fetch: bool = True,
) -> list[IclrDecisionRow]:
    """Walk each cell's candidates in their fixed (already hash-sorted) order, keeping the first
    `target_per_cell` that fetch+parse successfully. Returns the rows that made it in, in cell
    order -- the actual dev set, which may fall short of `len(candidates) * target_per_cell` if
    a cell's buffer runs out (logged, never silently padded from another cell)."""
    parser = OpenReviewParser(settings, fetch)
    picked: list[IclrDecisionRow] = []
    try:
        for cell, rows in candidates.items():
            kept = 0
            for row in rows:
                if kept >= target_per_cell:
                    break
                paper = parser.get(to_submission(row))
                if paper is None:
                    logger.warning(
                        "dev set %s: %s failed to fetch/parse, trying the next candidate",
                        cell,
                        row.submission_id,
                    )
                    continue
                register_parsed(settings, paper)
                picked.append(row)
                kept += 1
            if kept < target_per_cell:
                logger.warning(
                    "dev set %s: only %d/%d fetched (candidate buffer exhausted)",
                    cell,
                    kept,
                    target_per_cell,
                )
    finally:
        parser.close()
    return picked
