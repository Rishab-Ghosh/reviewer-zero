"""`reviewer eval` implementation. Owned by the eval workstream."""

import logging
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import typer

from reviewer.config import get_settings
from reviewer.models import Failure, PaperTier

from .annotate import annotate_records, public_records
from .grades import build_grades_template, load_grades_file, render_summary, summarize, write_grades
from .loader import (
    OWN_PREFIX,
    build_records,
    latest_failures,
    load_all_gold,
    load_claims,
    load_corpus,
    load_failures_tolerant,
    load_parsed,
)
from .parse_quality import parse_quality_row, render_parse_quality
from .report import render_report
from .schemas import PaperRecord

logger = logging.getLogger(__name__)

REPORT_NAME = "week1_claims.md"
GRADES_NAME = "grades.yaml"


def load_records() -> tuple[list[PaperRecord], list[Failure]]:
    """Load every stage's output from disk into ordered records (not yet annotated)."""
    s = get_settings()
    corpus = load_corpus(s.corpus_file)
    parsed, parse_errors = load_parsed(s.parsed_dir)
    claims, claims_errors = load_claims(s.claims_dir)
    failures = load_failures_tolerant(s.failures_file)
    known = (
        {e.paper_id for e in corpus}
        | set(parsed)
        | set(claims)
        | {f.paper_id for f in failures if f.paper_id.startswith(OWN_PREFIX)}
    )
    gold, private_ids = load_all_gold(s.gold_dir, s.private_gold_dir, known)
    load_errors: dict[str, list[str]] = defaultdict(list)
    for errs in (parse_errors, claims_errors):
        for pid, msg in errs.items():
            load_errors[pid].append(msg)
    records = build_records(
        corpus, parsed, claims, gold, failures, dict(load_errors), private_gold_ids=private_ids
    )
    return records, latest_failures(failures)


def public_failures(failures: list[Failure], records: list[PaperRecord]) -> list[Failure]:
    """Failures of papers that appear in the public report."""
    public_ids = {r.entry.paper_id for r in records}
    return [f for f in failures if f.paper_id in public_ids]


def run_eval() -> None:
    """Write the private grading report + grades file, and the public report.

    - reports/private/week1_claims.md: everything (own papers, private gold). Gitignored.
    - reports/private/grades.yaml: grades for every claim (filled grades are never lost).
    - reports/week1_claims.md: no own papers, no private gold. Safe to commit.
    """
    s = get_settings()
    threshold = s.duplicate_threshold
    now = datetime.now(UTC)
    raw, failures = load_records()
    records = annotate_records(raw, threshold)

    s.private_reports_dir.mkdir(parents=True, exist_ok=True)
    private_path = s.private_reports_dir / REPORT_NAME
    private_path.write_text(render_report(records, failures, now, threshold), encoding="utf-8")

    pub = annotate_records(public_records(raw), threshold)
    public_path = s.reports_dir / REPORT_NAME
    public_path.write_text(
        render_report(pub, public_failures(failures, pub), now, threshold, public=True),
        encoding="utf-8",
    )

    tiers = Counter(r.entry.tier.value for r in records)
    flag_kinds = Counter(f.kind.value for r in records for f in r.flags)
    logger.info(
        "wrote %s: %d papers (%s), %d parsed, %d with claims, %d claims, %d with gold (%d private)",
        private_path,
        len(records),
        ", ".join(f"{t}={tiers.get(t.value, 0)}" for t in PaperTier),
        sum(r.parsed is not None for r in records),
        sum(r.claims is not None for r in records),
        sum(len(r.claims.claims) for r in records if r.claims is not None),
        sum(r.gold is not None for r in records),
        sum(r.gold is not None and r.gold_private for r in records),
    )
    logger.info(
        "flags: %d total (%s); %d failures recorded",
        sum(flag_kinds.values()),
        ", ".join(f"{k}={v}" for k, v in sorted(flag_kinds.items())) or "none",
        len(failures),
    )
    logger.info("wrote %s: %d papers (own papers and private gold omitted)", public_path, len(pub))

    grades_path = s.private_reports_dir / GRADES_NAME
    legacy = s.reports_dir / GRADES_NAME
    if legacy.exists():
        logger.warning("ignoring old %s; the grading file is now %s", legacy, grades_path)
    gold_ids = [r.entry.paper_id for r in records if r.gold is not None]
    try:
        write_grades(grades_path, build_grades_template(records), gold_ids)
    except ValueError as e:
        logger.error("not touching %s because it could not be read: %s", grades_path, e)
        raise typer.Exit(code=1) from e


def run_summarize(grades_path: Path | None = None) -> None:
    """Print average scores by contribution type and by tier."""
    s = get_settings()
    path = grades_path or s.private_reports_dir / GRADES_NAME
    if not path.exists():
        typer.echo(f"no grades yet: {path} does not exist (run `reviewer eval` first)")
        return
    try:
        grades = load_grades_file(path).grades
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from e
    gold, _ = load_all_gold(s.gold_dir, s.private_gold_dir)
    table = summarize(grades, gold.keys())
    if table.n_graded == 0:
        typer.echo(
            f"no grades yet: 0/{table.n_total} claims graded across {table.n_papers} papers "
            f"({path})"
        )
        return
    typer.echo(render_summary(table))


def run_parse_quality() -> None:
    """Print lost tables and suspicious authors for every parsed paper (terminal only)."""
    s = get_settings()
    parsed, errors = load_parsed(s.parsed_dir)
    tiers = {e.paper_id: e.tier.value for e in load_corpus(s.corpus_file)}
    rows = [
        parse_quality_row(p, tiers.get(pid, "own" if pid.startswith(OWN_PREFIX) else "?"))
        for pid, p in parsed.items()
    ]
    typer.echo(render_parse_quality(rows))
    for msg in errors.values():
        typer.echo(f"unreadable: {msg}", err=True)
