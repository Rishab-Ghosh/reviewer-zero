"""`reviewer review PDF --out review.json` (task 1.9): one review with the web profile."""

import json
import logging
import shutil
from pathlib import Path

import typer

from reviewer.cache import sha256_file
from reviewer.config import get_settings
from reviewer.models import StepStatus
from reviewer.parse.ids import paper_id_for
from reviewer.privacy import write_private
from reviewer.review.budget import Budget
from reviewer.review.orchestrator import VersionContext, run_review
from reviewer.review.profile import init_work_dir, paper_settings

logger = logging.getLogger(__name__)


def review_pdf(
    pdf: Path,
    out: Path,
    work_dir: Path | None,
    venue: str,
    venue_year: int | None,
    limit_usd: float,
) -> dict:
    base = get_settings()
    work = (work_dir or out.parent / f"{out.stem}.work").resolve()
    s = paper_settings(base, work, public_cache=base.cache_dir, paper_cache=work / "cache")
    init_work_dir(s)
    local_pdf = work / "pdf" / pdf.name
    local_pdf.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pdf, local_pdf)
    paper_id = paper_id_for(local_pdf, sha256_file(local_pdf))
    ctx = VersionContext(
        paper_id=paper_id, settings=s, venue=venue, venue_year=venue_year, pdf=local_pdf
    )
    budget = Budget(limit_usd=limit_usd)
    results = run_review(ctx, budget)
    outputs = {
        r.step.value: json.loads(Path(r.output).read_text(encoding="utf-8"))
        for r in results
        if r.output and r.status == StepStatus.done
    }
    body = {
        "paper_id": paper_id,
        "limit_usd": limit_usd,
        "spent_usd": round(budget.spent_usd, 4),
        "steps": [r.model_dump(mode="json", exclude={"output"}) for r in results],
        "outputs": outputs,
    }
    write_private(out, json.dumps(body, indent=2, default=str) + "\n")
    for r in results:
        typer.echo(f"{r.step.value:10} {r.status.value:10} ${r.cost_usd:.4f} {r.error_kind or ''}")
    typer.echo(f"spent ${budget.spent_usd:.2f} of ${limit_usd:.2f} -> {out}")
    return body
