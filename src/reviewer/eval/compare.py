"""Side-by-side comparison of claims from the default and the strong model."""

import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from reviewer.config import Settings, get_settings
from reviewer.eval.checks import build_idf, run_checks
from reviewer.eval.loader import load_claims, load_parsed
from reviewer.models import Claim, PaperClaims, ParsedPaper

logger = logging.getLogger(__name__)

DEFAULT_COMPARE_IDS = ("2601.19871", "2512.22069", "2609.04881", "1706.03762", "2305.18290")


class ComparisonRow(BaseModel):
    paper_id: str
    title: str
    paper: ParsedPaper | None
    default: PaperClaims | None
    strong: PaperClaims | None


def _cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def _cost(pc: PaperClaims | None) -> str:
    if pc is None:
        return "—"
    u = pc.extraction_metadata.token_usage
    tokens_in = u.input_tokens + u.cache_read_input_tokens + u.cache_creation_input_tokens
    return f"{tokens_in:,} in / {u.output_tokens:,} out, ${u.cost_usd:.4f}"


def _flags(pc: PaperClaims | None, paper: ParsedPaper | None, idf: dict[str, float]) -> str:
    if pc is None:
        return "—"
    flags = run_checks(pc, paper, idf)
    return str(len(flags)) + (
        f" ({', '.join(sorted({f.kind.value for f in flags}))})" if flags else ""
    )


def _claim_details(c: Claim) -> str:
    return (
        f"**{c.contribution_type.value}** (conf {c.confidence:.2f})<br>"
        f"**method:** {_cell(c.method)}<br>**setting:** {_cell(c.setting)}<br>"
        f"**result:** {_cell(c.result)}<br>**novelty:** {_cell(c.novelty_assertion)}<br>"
        f"**evidence:** {_cell(', '.join(c.evidence))}"
    )


def render_comparison(
    rows: Sequence[ComparisonRow], default_model: str, strong_model: str, now: datetime
) -> str:
    all_statements = [
        c.statement for r in rows for pc in (r.default, r.strong) if pc for c in pc.claims
    ]
    idf = build_idf(all_statements)
    versions = sorted(
        {pc.extraction_metadata.prompt_version for r in rows for pc in (r.default, r.strong) if pc}
    )
    version_text = ", ".join(f"`{v}`" for v in versions) or "unknown"
    lines = [
        "# Model comparison: claim extraction",
        "",
        f"Generated {now:%Y-%m-%d %H:%M} UTC. Same parsed input and effort; prompt "
        f"version(s) in these outputs: {version_text}. Only the model differs.",
        f"Default model: `{default_model}`. Strong model: `{strong_model}`.",
        "Claims are aligned by position only (claim k of each model), not by topic.",
        "",
        "| paper | id | default: claims · flags · tokens/cost | "
        "strong: claims · flags · tokens/cost |",
        "|---|---|---|---|",
    ]
    for r in rows:
        d = f"{len(r.default.claims) if r.default else '—'} · {_flags(r.default, r.paper, idf)}"
        s = f"{len(r.strong.claims) if r.strong else '—'} · {_flags(r.strong, r.paper, idf)}"
        lines.append(
            f"| {_cell(r.title)[:60]} | `{r.paper_id}` | {d} · {_cost(r.default)} | "
            f"{s} · {_cost(r.strong)} |"
        )
    for r in rows:
        lines += ["", f"## {_cell(r.title)} (`{r.paper_id}`)", ""]
        for label, pc in ((default_model, r.default), (strong_model, r.strong)):
            summary = _cell(pc.one_line_summary) if pc else "_no output_"
            lines.append(f"- **{label} summary:** {summary}")
        lines += ["", f"| # | {default_model} | {strong_model} |", "|---|---|---|"]
        n = max(len(r.default.claims) if r.default else 0, len(r.strong.claims) if r.strong else 0)
        for i in range(n):
            cells = []
            for pc in (r.default, r.strong):
                if pc and i < len(pc.claims):
                    c = pc.claims[i]
                    cells.append(f"**{_cell(c.statement)}**<br><br>{_claim_details(c)}")
                else:
                    cells.append("—")
            lines.append(f"| {i + 1} | {cells[0]} | {cells[1]} |")
        if r.default is None or r.strong is None:
            missing = default_model if r.default is None else strong_model
            lines += ["", f"_No claims from `{missing}` for this paper (see data/failures.jsonl)._"]
    return "\n".join(lines) + "\n"


def run_compare(
    ids: list[str] | None = None,
    strong_model: str | None = None,
    force: bool = False,
    settings: Settings | None = None,
) -> Path:
    """Extract `ids` with the strong model into data/claims_strong/ (cached), then write
    reports/model_comparison.md next to the default-model claims in data/claims/."""
    from reviewer.claims.commands import run_extract

    settings = settings or get_settings()
    ids = list(ids or DEFAULT_COMPARE_IDS)
    strong_model = strong_model or settings.llm_model_strong
    strong_dir = settings.data_dir / "claims_strong"
    strong_dir.mkdir(parents=True, exist_ok=True)
    run_extract(model=strong_model, ids=ids, out_dir=strong_dir, force=force)

    parsed, _ = load_parsed(settings.parsed_dir)
    default_claims, _ = load_claims(settings.claims_dir)
    strong_claims, _ = load_claims(strong_dir)
    rows = [
        ComparisonRow(
            paper_id=pid,
            title=parsed[pid].title if pid in parsed else pid,
            paper=parsed.get(pid),
            default=default_claims.get(pid),
            strong=strong_claims.get(pid),
        )
        for pid in ids
    ]
    default_model = next(
        (r.default.extraction_metadata.model for r in rows if r.default), settings.llm_model
    )
    # Own-paper content must never reach a committed file.
    has_own = any(pid.startswith("own-") for pid in ids)
    out_dir = settings.private_reports_dir if has_own else settings.reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "model_comparison.md"
    if has_own:
        logger.info("comparison includes an own paper; writing to private %s", out)
    out.write_text(
        render_comparison(rows, default_model, strong_model, datetime.now(UTC)), encoding="utf-8"
    )
    logger.info("wrote %s", out)
    return out
