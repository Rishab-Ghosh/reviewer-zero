"""The 60-paper ICLR dev run (plan §10): `reviewer meta dev-fetch` (sample + download + parse,
no LLM) and `reviewer meta dev-run` (every checker, lite, + meta judgment + baselines + probe
under the spend cap; then the 5-fold CV calibration and the report). All outputs private."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import typer

from reviewer.config import Settings, get_settings
from reviewer.meta.calibration import extract_features
from reviewer.meta.commands import BudgetExceededError, run_batch
from reviewer.meta.dev_report import build_report
from reviewer.meta.devset import eligible, load_decisions, mean_rating, stratified_sample
from reviewer.meta.ingest import fetch_dev_set
from reviewer.models import CheckerRunStatus, DevRow, IclrDecisionRow, MetaReview
from reviewer.novelty.commands import git_state
from reviewer.privacy import append_private, write_private

logger = logging.getLogger(__name__)

VENUES = ("ICLR.cc/2024/Conference", "ICLR.cc/2025/Conference")
STATUSES = ("accepted", "rejected", "withdrawn")
SEED_TAG = "meta-dev-v1"
PER_CELL = 10
BUFFER = 8
RUNS_LOG = "meta_dev_runs.jsonl"


def dev_set_path(settings: Settings) -> Path:
    return settings.data_dir / "openreview" / "meta_dev_set.jsonl"


def paper_id(row: IclrDecisionRow) -> str:
    return f"or-{row.submission_id}"


def load_dev_set(settings: Settings) -> list[IclrDecisionRow]:
    path = dev_set_path(settings)
    if not path.exists():
        logger.error("no dev set at %s: run `reviewer meta dev-fetch` first", path)
        raise typer.Exit(code=1)
    return load_decisions(path)


def run_dev_fetch(settings: Settings | None = None) -> list[IclrDecisionRow]:
    """Sample 10 per (year, decision) cell from decisions.jsonl, download each PDF as
    reviewers saw it and parse it with GROBID, skipping to the cell's next candidate on failure.
    Free (no LLM); slow (OpenReview throttling)."""
    settings = settings or get_settings()
    rows = eligible(load_decisions(settings.meta_decisions_file), VENUES)
    cells = [(v, s) for v in VENUES for s in STATUSES]
    candidates = stratified_sample(rows, cells, PER_CELL, SEED_TAG, BUFFER)
    picked = fetch_dev_set(candidates, PER_CELL, settings)
    path = dev_set_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(r.model_dump_json() + "\n" for r in picked), encoding="utf-8")
    typer.echo(f"dev set: {len(picked)}/{len(cells) * PER_CELL} papers parsed -> {path}")
    return picked


def dev_row(row: IclrDecisionRow, review: MetaReview) -> DevRow:
    probe = review.contamination_probe
    return DevRow(
        paper_id=review.paper_id,
        venue=row.venue,
        year=row.year,
        final_status=row.final_status,
        mean_reviewer_rating=mean_rating(row) or 0.0,
        reviewer_ratings=row.ratings,
        meta_raw_rating=extract_features(review.judgments)["raw_rating"],
        baseline_abstract_rating=(
            review.baseline_abstract.raw_rating if review.baseline_abstract else None
        ),
        baseline_full_rating=review.baseline_full.raw_rating if review.baseline_full else None,
        contamination_confidence=probe.confidence if probe else None,
        contamination_accepted_guess=probe.accepted_guess if probe else None,
        checkers_ok=sum(s.status == CheckerRunStatus.ok for s in review.checker_status),
        checkers_total=len(review.checker_status),
    )


def interleave_cells(rows: list[IclrDecisionRow]) -> list[IclrDecisionRow]:
    """Round-robin over (year, decision) cells, keeping each cell's order: if the spend cap stops
    the batch early, every cell is still (almost) equally represented."""
    cells: dict[tuple[int, str], list[IclrDecisionRow]] = {}
    for r in rows:
        cells.setdefault((r.year, r.final_status), []).append(r)
    queues = list(cells.values())
    out: list[IclrDecisionRow] = []
    for i in range(max((len(q) for q in queues), default=0)):
        out.extend(q[i] for q in queues if i < len(q))
    return out


def run_dev_run(settings: Settings | None = None, rejudge_from: str | None = None) -> dict:
    """`rejudge_from`: judge only the papers that have a review under that earlier meta prompt
    (their checker outputs are reused), so a new prompt is compared on exactly the same set."""
    settings = settings or get_settings()
    settings = settings.model_copy(update={"meta_blind_decision_status": True})
    commit, dirty = git_state()
    dev = interleave_cells(load_dev_set(settings))
    ids = [paper_id(r) for r in dev]
    if rejudge_from:
        prev = settings.private_reports_dir / "meta" / rejudge_from
        ids = [pid for pid in ids if (prev / f"{pid}.json").exists()]
    stopped = None
    try:
        run_batch(
            ids,
            venue="ICLR",
            lite=True,
            settings=settings,
            venue_years={paper_id(r): r.year for r in dev},
        )
    except BudgetExceededError as e:
        stopped = str(e)
        logger.error("dev run stopped: %s", e)

    reviews: dict[str, MetaReview] = {}
    for pid in ids:
        path = settings.meta_dir / f"{pid}.json"
        if path.exists():
            reviews[pid] = MetaReview.model_validate_json(path.read_text(encoding="utf-8"))
    rows = [dev_row(r, reviews[paper_id(r)]) for r in dev if paper_id(r) in reviews]
    report = build_report(rows)
    spend = sum(rv.token_usage.cost_usd for rv in reviews.values())
    run = {
        "type": "meta_dev_run",
        "timestamp": datetime.now(UTC).isoformat(),
        "split": "dev",
        "commit": commit,
        "dirty": dirty,
        "n_dev_set": len(dev),
        "rejudge_from": rejudge_from,
        "n_reviewed": len(rows),
        "not_reviewed": [
            {"paper_id": paper_id(r), "year": r.year, "final_status": r.final_status}
            for r in dev
            if paper_id(r) not in reviews
        ],
        "stopped": stopped,
        "meta_judgment_and_baselines_cost_usd": spend,
        "config": {
            "meta_model": settings.meta_model,
            "meta_prompt": settings.meta_prompt,
            "meta_k": settings.meta_k,
            "novelty_top_claims": settings.meta_novelty_top_claims,
            "budget_usd": settings.meta_budget_usd,
            "blind_decision_status": True,
        },
        "report": report,
    }
    out = settings.private_reports_dir
    write_private(
        out / f"meta_dev_run_{settings.meta_prompt}.json",
        json.dumps({**run, "rows": [r.model_dump() for r in rows]}, indent=2, default=str) + "\n",
    )
    append_private(out / RUNS_LOG, json.dumps(run, default=str))
    typer.echo(
        f"meta dev run @ {commit[:10]}{' (dirty)' if dirty else ''}: {len(rows)}/{len(dev)} "
        f"papers; Spearman raw {report['correlation']['meta_raw']:.3f}, "  # type: ignore[index]
        f"AUC raw {report['auc']['meta_raw']:.3f}" + (f"; STOPPED: {stopped}" if stopped else "")  # type: ignore[index]
    )
    return run
