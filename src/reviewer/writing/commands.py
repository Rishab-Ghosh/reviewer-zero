"""`reviewer writing PATH` and `reviewer writing-summary`."""

import logging
from pathlib import Path

import typer

from reviewer.config import Settings, get_settings
from reviewer.failures import record_failure
from reviewer.llm.client import LLMClient, LLMError, MissingAPIKeyError
from reviewer.models import ParsedPaper, WritingReport
from reviewer.privacy import write_private
from reviewer.writing.checker import WritingError, check_paper, load_writing_prompt
from reviewer.writing.report import render_report, summary_markdown

logger = logging.getLogger(__name__)


def is_private(paper: ParsedPaper) -> bool:
    src = paper.source_path.replace("\\", "/")
    return paper.paper_id.startswith(("own-", "or-")) or "papers/own/" in src or "/private/" in src


def report_dir(settings: Settings, private: bool) -> Path:
    return (settings.private_reports_dir if private else settings.reports_dir) / "writing"


def _write(path: Path, text: str, private: bool) -> None:
    if private:
        write_private(path, text)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_report(report: WritingReport, settings: Settings) -> Path:
    out_dir = report_dir(settings, report.private)
    md = out_dir / f"{report.paper_id}.md"
    _write(md, render_report(report), report.private)
    _write(md.with_suffix(".json"), report.model_dump_json(indent=2) + "\n", report.private)
    return md


def run_writing(
    path: Path,
    force: bool = False,
    settings: Settings | None = None,
    client: LLMClient | None = None,
) -> None:
    settings = settings or get_settings()
    client = client or LLMClient(settings)
    prompt = load_writing_prompt(settings.writing_prompt)
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    for f in files:
        paper = ParsedPaper.model_validate_json(f.read_text(encoding="utf-8"))
        private = is_private(paper)
        out = report_dir(settings, private) / f"{paper.paper_id}.json"
        if out.exists() and not force:
            logger.info("%s: report exists, skipping (use --force)", paper.paper_id)
            continue
        if client.totals.fresh.cost_usd >= settings.writing_budget_usd:
            logger.error(
                "budget $%.2f reached ($%.2f spent); stopping before %s",
                settings.writing_budget_usd,
                client.totals.fresh.cost_usd,
                paper.paper_id,
            )
            break
        try:
            report = check_paper(
                client,
                paper,
                private=private,
                prompt=prompt,
                model=settings.writing_model,
                max_retries=settings.llm_max_retries,
            )
        except MissingAPIKeyError as e:
            logger.error("%s", e)
            raise typer.Exit(code=1) from e
        except (WritingError, LLMError, OSError, ValueError) as e:
            record_failure(
                settings.failures_file, "writing", paper.paper_id, f"{type(e).__name__}: {e}"
            )
            continue
        md = write_report(report, settings)
        typer.echo(
            f"{paper.paper_id}: {len(report.findings)} findings, {len(report.dropped)} dropped, "
            f"${report.token_usage.cost_usd:.2f} -> {md}"
        )
    typer.echo(
        f"new spend ${client.totals.fresh.cost_usd:.2f} ({client.totals.n_fresh} calls, "
        f"{client.totals.n_cached} cached)"
    )


def run_summary(seed: int, n_sample: int, settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    for private in (False, True):
        d = report_dir(settings, private)
        paths = sorted(d.glob("*.json")) if d.is_dir() else []
        reports = [WritingReport.model_validate_json(p.read_text(encoding="utf-8")) for p in paths]
        if not reports:
            continue
        out = d / "SUMMARY.md"
        _write(out, summary_markdown(reports, seed, n_sample), private)
        typer.echo(f"{len(reports)} report(s) -> {out}")
