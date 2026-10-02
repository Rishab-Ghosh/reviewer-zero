"""`reviewer methods ...`: methodology checker, its evaluation against critiques, and the
precision sample. All outputs are private (reports/private/): they cover own and lab papers,
and the evaluation quotes critiques."""

import logging
from pathlib import Path

import typer
import yaml

from reviewer.config import Settings, get_settings
from reviewer.corpus.corpus import load_corpus
from reviewer.failures import record_failure
from reviewer.llm.client import LLMClient, LLMError, MissingAPIKeyError
from reviewer.methods.checker import MethodsError, check_paper, load_methods_prompt
from reviewer.methods.consensus import check_paper_twice
from reviewer.methods.evaluate import (
    CritiqueOutcome,
    Raised,
    match_paper,
    outcomes,
    precision_sample,
    verify_paper,
)
from reviewer.methods.report import render_eval, render_precision_grading, render_reviews
from reviewer.models import ConcernCheck, Critiques, PaperMethodsReview, ParsedPaper
from reviewer.privacy import write_private

logger = logging.getLogger(__name__)

REPAIR_PROMPT = "repair_v1"
MATCH_PROMPT = "match_v1"
VERIFY_PROMPT = "verify_v1"


def methods_client(settings: Settings) -> LLMClient:
    return LLMClient(settings.model_copy(update={"llm_effort": settings.methods_effort}))


def _parsed(settings: Settings, paper_id: str) -> ParsedPaper:
    path = settings.parsed_dir / f"{paper_id}.json"
    return ParsedPaper.model_validate_json(path.read_text(encoding="utf-8"))


def load_reviews(settings: Settings) -> list[PaperMethodsReview]:
    return [
        PaperMethodsReview.model_validate_json(p.read_text(encoding="utf-8"))
        for p in sorted(settings.methods_dir.glob("*.json"))
    ]


def load_critiques(settings: Settings) -> list[Critiques]:
    return [
        Critiques.model_validate(yaml.safe_load(p.read_text(encoding="utf-8")))
        for p in sorted(settings.critiques_dir.glob("*.yaml"))
    ]


def _write_report(settings: Settings, reviews: list[PaperMethodsReview]) -> Path:
    titles = {e.paper_id: e.title or "" for e in load_corpus(settings)}
    for r in reviews:
        if not titles.get(r.paper_id):
            titles[r.paper_id] = _parsed(settings, r.paper_id).title
    out = settings.private_reports_dir / "methods_report.md"
    write_private(out, render_reviews(reviews, titles))
    return out


def run_check(
    ids: list[str] | None = None,
    force: bool = False,
    settings: Settings | None = None,
    client: LLMClient | None = None,
) -> list[PaperMethodsReview]:
    """Review every parsed corpus paper (or `ids`); METHODS_RUNS (default 2) Opus calls per
    paper (+ a repair call when quotes fail), cached; two runs give confirmed/possible."""
    settings = settings or get_settings()
    if settings.methods_runs not in (1, 2):
        raise typer.BadParameter("METHODS_RUNS must be 1 or 2")
    tiers = {e.paper_id: e.tier.value for e in load_corpus(settings)}
    paper_ids = ids or sorted(p.stem for p in settings.parsed_dir.glob("*.json") if p.stem in tiers)
    client = client or methods_client(settings)
    prompt = load_methods_prompt(settings.methods_prompt)
    repair = load_methods_prompt(REPAIR_PROMPT) if settings.methods_repair else None
    results: list[PaperMethodsReview] = []
    for pid in paper_ids:
        out = settings.methods_dir / f"{pid}.json"
        if out.exists() and not force:
            results.append(PaperMethodsReview.model_validate_json(out.read_text(encoding="utf-8")))
            continue
        try:
            check = check_paper_twice if settings.methods_runs == 2 else check_paper
            result = check(
                client,
                _parsed(settings, pid),
                tier=tiers.get(pid, "unknown"),
                prompt=prompt,
                repair_prompt=repair,
                model=settings.methods_model,
                effort=settings.methods_effort,
                max_retries=settings.llm_max_retries,
            )
        except MissingAPIKeyError as e:
            logger.error("%s", e)
            raise typer.Exit(code=1) from e
        except (MethodsError, LLMError, OSError, ValueError) as e:
            record_failure(settings.failures_file, "methods", pid, f"{type(e).__name__}: {e}")
            continue
        write_private(out, result.model_dump_json(indent=2) + "\n")
        results.append(result)
        logger.info(
            "methods %s: $%.2f; downgraded %d; spend so far $%.2f",
            pid,
            result.token_usage.cost_usd,
            sum(f.downgraded for f in result.findings),
            client.totals.fresh.cost_usd,
        )
    logger.info(
        "methods: %d/%d papers; new spend $%.2f (%d calls, %d cached)",
        len(results),
        len(paper_ids),
        client.totals.fresh.cost_usd,
        client.totals.n_fresh,
        client.totals.n_cached,
    )
    if results:
        report = _write_report(settings, load_reviews(settings))
        typer.echo(f"reviewed {len(results)}/{len(paper_ids)} papers; wrote {report}")
    return results


def run_eval(
    settings: Settings | None = None,
    client: LLMClient | None = None,
    precision_n: int = 20,
) -> tuple[list[CritiqueOutcome], list[tuple[Raised, ConcernCheck]]]:
    """Recall against every critiques file, plus an audited precision sample; writes
    reports/private/methods_eval.md and methods_precision_sample.md."""
    settings = settings or get_settings()
    client = client or methods_client(settings)
    reviews = {r.paper_id: r for r in load_reviews(settings)}
    if not reviews:
        logger.error("no reviews in %s; run `reviewer methods check`", settings.methods_dir)
        raise typer.Exit(code=1)
    match_prompt = load_methods_prompt(MATCH_PROMPT)
    items: list[CritiqueOutcome] = []
    missing: list[str] = []
    for crit in load_critiques(settings):
        review = reviews.get(crit.paper_id)
        if review is None:
            missing.append(crit.paper_id)
            continue
        try:
            matches, _ = match_paper(
                client,
                crit,
                review,
                prompt=match_prompt,
                model=settings.methods_model,
                max_retries=settings.llm_max_retries,
            )
        except (MethodsError, LLMError) as e:
            record_failure(
                settings.failures_file, "methods", crit.paper_id, f"match: {type(e).__name__}: {e}"
            )
            continue
        items += outcomes(crit, review, matches)

    verify_prompt = load_methods_prompt(VERIFY_PROMPT)
    sample = precision_sample(list(reviews.values()), precision_n)
    write_private(
        settings.private_reports_dir / "methods_precision_sample.md",
        render_precision_grading(sample),
    )
    checks: list[tuple[Raised, ConcernCheck]] = []
    for pid in sorted({r.paper_id for r in sample}):
        concerns = [r for r in sample if r.paper_id == pid]
        try:
            got, _ = verify_paper(
                client,
                _parsed(settings, pid),
                concerns,
                prompt=verify_prompt,
                model=settings.methods_model,
                max_retries=settings.llm_max_retries,
            )
        except (MethodsError, LLMError) as e:
            record_failure(
                settings.failures_file, "methods", pid, f"verify: {type(e).__name__}: {e}"
            )
            continue
        by_id = {c.concern_id: c for c in got}
        checks += [(r, by_id[r.key]) for r in concerns]

    out = settings.private_reports_dir / "methods_eval.md"
    write_private(out, render_eval(items, checks, missing, client.totals.fresh.cost_usd))
    logger.info(
        "methods eval: new spend $%.2f (%d calls, %d cached)",
        client.totals.fresh.cost_usd,
        client.totals.n_fresh,
        client.totals.n_cached,
    )
    typer.echo(f"{len(items)} critiques matched, {len(checks)} concerns audited; wrote {out}")
    return items, checks


def run_report(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    reviews = load_reviews(settings)
    if not reviews:
        logger.error("no reviews in %s", settings.methods_dir)
        raise typer.Exit(code=1)
    typer.echo(f"wrote {_write_report(settings, reviews)}")
