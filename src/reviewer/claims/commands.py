"""`reviewer extract` implementation. Owned by the claims workstream."""

import logging
from pathlib import Path

import typer

from reviewer.claims.extractor import ExtractionError, extract_claims
from reviewer.config import Settings, get_settings
from reviewer.failures import record_failure
from reviewer.llm.client import LLMClient, LLMError, MissingAPIKeyError, UsageTotals
from reviewer.models import ParsedPaper

logger = logging.getLogger(__name__)


def parsed_paths(settings: Settings, ids: list[str] | None) -> list[Path]:
    if ids:
        return [settings.parsed_dir / f"{paper_id}.json" for paper_id in ids]
    return sorted(settings.parsed_dir.glob("*.json"))


def write_json(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text + "\n", encoding="utf-8")
    tmp.replace(path)


def extract_one(
    path: Path, model: str, out_dir: Path, force: bool, client: LLMClient, settings: Settings
) -> bool:
    """Extract one paper. Returns True on success or skip; failures are recorded."""
    paper_id = path.stem
    out_path = out_dir / f"{paper_id}.json"
    if out_path.exists() and not force:
        logger.info("skip %s: %s exists (use --force)", paper_id, out_path)
        return True
    if not path.exists():
        record_failure(settings.failures_file, "extract", paper_id, f"no parsed file {path}")
        return False
    try:
        paper = ParsedPaper.model_validate_json(path.read_text(encoding="utf-8"))
        claims, draft = extract_claims(paper, model, client, settings)
    except MissingAPIKeyError:
        raise
    except ExtractionError as e:
        if e.pass1 is not None:
            write_json(out_dir / f"{paper_id}.pass1.json", e.pass1.model_dump_json(indent=2))
        record_failure(settings.failures_file, "extract", paper_id, f"[{model}] {e}")
        return False
    except (LLMError, ValueError, OSError) as e:
        record_failure(
            settings.failures_file, "extract", paper_id, f"[{model}] {type(e).__name__}: {e}"
        )
        return False
    write_json(out_dir / f"{paper_id}.pass1.json", draft.model_dump_json(indent=2))
    write_json(out_path, claims.model_dump_json(indent=2))
    logger.info("wrote %s", out_path)
    return True


def log_totals(totals: UsageTotals) -> None:
    for label, usage, n in (
        ("fresh", totals.fresh, totals.n_fresh),
        ("cached", totals.cached, totals.n_cached),
    ):
        logger.info(
            "%s LLM calls: %d, %d in / %d out tokens (+%d cache-read, %d cache-write), $%.4f",
            label,
            n,
            usage.input_tokens,
            usage.output_tokens,
            usage.cache_read_input_tokens,
            usage.cache_creation_input_tokens,
            usage.cost_usd,
        )
    logger.info(
        "total cost $%.4f (new spend $%.4f, cached originals $%.4f)",
        totals.fresh.cost_usd + totals.cached.cost_usd,
        totals.fresh.cost_usd,
        totals.cached.cost_usd,
    )


def run_extract(
    model: str | None = None,
    ids: list[str] | None = None,
    out_dir: Path | None = None,
    force: bool = False,
    settings: Settings | None = None,
    client: LLMClient | None = None,
) -> None:
    """Extract claims for every data/parsed/*.json (or just `ids`) into `out_dir`
    (default data/claims/). `model` defaults to settings.llm_model."""
    settings = settings or get_settings()
    model = model or settings.llm_model
    out_dir = out_dir or settings.claims_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    client = client or LLMClient(settings)

    paths = parsed_paths(settings, ids)
    logger.info("extracting %d papers with %s into %s", len(paths), model, out_dir)
    n_ok = 0
    try:
        for path in paths:
            n_ok += extract_one(path, model, out_dir, force, client, settings)
    except MissingAPIKeyError as e:
        logger.error("%s", e)
        log_totals(client.totals)
        raise typer.Exit(code=1) from e
    logger.info("extract: %d/%d papers ok or skipped", n_ok, len(paths))
    log_totals(client.totals)
