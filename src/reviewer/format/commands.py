"""`reviewer format` and `reviewer format-summary`."""

import logging
import random
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import typer

from reviewer.config import PROJECT_ROOT, Settings, get_settings
from reviewer.failures import record_failure
from reviewer.format.checks import Context, run_all
from reviewer.format.layout import declared_venue, find_headings, main_text_pages, references_span
from reviewer.format.pdf import PdfDoc, TextEngine, load_pdf
from reviewer.format.report import render_report
from reviewer.format.venues import VENUES
from reviewer.models import (
    DeclaredVenue,
    FormatCheck,
    FormatReport,
    FormatSeverity,
    ParsedPaper,
)
from reviewer.parse.pages import PdfToTextError

logger = logging.getLogger(__name__)


def is_private(paper: ParsedPaper) -> bool:
    src = paper.source_path.replace("\\", "/")
    return paper.paper_id.startswith(("own-", "or-")) or "papers/own/" in src or "/private/" in src


def report_dir(settings: Settings, private: bool) -> Path:
    return (settings.private_reports_dir if private else settings.reports_dir) / "format"


def resolve_pdf(paper: ParsedPaper, pdf_root: Path) -> Path:
    src = Path(paper.source_path)
    return src if src.is_absolute() else pdf_root / src


def check_pdf(
    pdf: Path,
    *,
    paper_id: str,
    title: str,
    private: bool,
    author_names: list[str],
    declared: DeclaredVenue | None = None,
    rules_mode: Literal["declared", "none"] = "declared",
    engine: TextEngine | None = None,
) -> tuple[FormatReport, PdfDoc]:
    """The format check on one PDF. `declared` overrides the venue/stage printed on page 1
    (D18: the user's choice wins); the detected header is still reported."""
    doc = load_pdf(pdf, engine)
    headings = find_headings(doc.pages)
    detected = declared_venue(doc.pages[0] if doc.pages else "")
    used = declared or detected
    pages = [main_text_pages(doc.pages, headings, rules) for rules in VENUES.values()]
    ctx = Context(
        pages=doc.pages,
        metadata=doc.metadata,
        headings=headings,
        declared=used,
        refs_span=references_span(doc.pages, headings),
        author_names=author_names,
        rules_mode=rules_mode,
    )
    report = FormatReport(
        paper_id=paper_id,
        title=title,
        private=private,
        generated=datetime.now(UTC),
        n_pages=len(doc.pages),
        declared=used,
        pages=pages,
        findings=run_all(ctx, pages),
    )
    return report, doc


def check_paper(paper: ParsedPaper, pdf: Path) -> FormatReport:
    report, _doc = check_pdf(
        pdf,
        paper_id=paper.paper_id,
        title=paper.title,
        private=is_private(paper),
        author_names=[a.name for a in paper.authors],
    )
    return report


def write_report(report: FormatReport, settings: Settings) -> Path:
    out_dir = report_dir(settings, report.private)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / f"{report.paper_id}.md"
    md.write_text(render_report(report), encoding="utf-8")
    md.with_suffix(".json").write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return md


def run_format(path: Path, pdf_root: Path | None = None, settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    root = pdf_root or PROJECT_ROOT
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    for f in files:
        paper = ParsedPaper.model_validate_json(f.read_text(encoding="utf-8"))
        pdf = resolve_pdf(paper, root)
        if not pdf.exists():
            record_failure(settings.failures_file, "format", paper.paper_id, f"no PDF at {pdf}")
            continue
        try:
            report = check_paper(paper, pdf)
        except (PdfToTextError, OSError, ValueError) as e:
            record_failure(
                settings.failures_file, "format", paper.paper_id, f"{type(e).__name__}: {e}"
            )
            continue
        out = write_report(report, settings)
        sev = Counter(x.severity.value for x in report.findings)
        typer.echo(
            f"{paper.paper_id}: {report.declared.venue.value if report.declared.venue else '-'} "
            f"{report.declared.stage.value}; "
            + ", ".join(f"{k}={sev.get(k, 0)}" for k in ("violation", "warning", "info"))
            + f" -> {out}"
        )


# --------------------------------------------------------------------------- summary


def summary_markdown(reports: list[FormatReport], seed: int, n_sample: int) -> str:
    checks = list(FormatCheck)
    head = (
        "| paper | venue / stage | pages (ICLR/NeurIPS/ICML) | V | W | I | "
        + " | ".join(c.value for c in checks)
        + " |"
    )
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    totals: Counter[str] = Counter()
    for r in reports:
        by_check = Counter(f.check.value for f in r.findings)
        sev = Counter(f.severity.value for f in r.findings)
        totals.update(by_check)
        totals.update({f"sev:{k}": v for k, v in sev.items()})
        pages = "/".join(str(p.pages_used) for p in r.pages)
        venue = f"{r.declared.venue.value if r.declared.venue else '-'} {r.declared.stage.value}"
        lines.append(
            f"| {r.paper_id} | {venue} | {pages} | {sev.get('violation', 0)} | "
            f"{sev.get('warning', 0)} | {sev.get('info', 0)} | "
            + " | ".join(str(by_check.get(c.value, 0)) for c in checks)
            + " |"
        )
    lines.append(
        f"| **total** | | | **{totals['sev:violation']}** | **{totals['sev:warning']}** | "
        f"**{totals['sev:info']}** | " + " | ".join(f"**{totals[c.value]}**" for c in checks) + " |"
    )
    pool = [
        (r.paper_id, f)
        for r in reports
        for f in r.findings
        if f.severity in (FormatSeverity.violation, FormatSeverity.warning)
        or f.check
        not in (
            FormatCheck.required_section,
            FormatCheck.recommended_section,
            FormatCheck.page_limit,
        )
    ]
    rng = random.Random(seed)
    sample = rng.sample(pool, min(n_sample, len(pool)))
    kinds = Counter(f.check.value for _, f in pool)
    lines += [
        "",
        f"#### {len(sample)} random findings (seed {seed})",
        "",
        f"Drawn uniformly from {len(pool)} detections (every anonymization and reference "
        "finding, plus violations/warnings of page limits and sections; not-binding info rows "
        "for other venues excluded): "
        + ", ".join(f"{k} {v}" for k, v in kinds.most_common())
        + ".",
        "",
    ]
    for i, (pid, f) in enumerate(sample, 1):
        where = f" p{f.page}" if f.page else ""
        lines.append(f"{i}. **{pid}{where} — {f.check.value}** ({f.severity.value}): {f.detail}")
        lines += [f"   - {' '.join(e.split())[:300]}" for e in f.evidence]
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run_summary(seed: int, n_sample: int, settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    for private in (False, True):
        d = report_dir(settings, private)
        reports = (
            [
                FormatReport.model_validate_json(p.read_text(encoding="utf-8"))
                for p in sorted(d.glob("*.json"))
            ]
            if d.is_dir()
            else []
        )
        if not reports:
            continue
        out = d / "SUMMARY.md"
        out.write_text(summary_markdown(reports, seed, n_sample), encoding="utf-8")
        typer.echo(f"{len(reports)} report(s) -> {out}")
