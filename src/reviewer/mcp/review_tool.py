"""review_paper: the full pipeline on the user's machine with the user's key (MCP_PLAN §3.4).

Two phases: confirm=false spends nothing and returns the estimate; confirm=true runs. Resume is
"the cheap way" (founder, Q-M3): the work dir and LLM cache live under
REVIEWER_ZERO_CACHE/papers/<pdf sha256>/, so a second call on the same PDF re-serves finished
calls from the cache and pays only for what is left.
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from reviewer.cache import sha256_file
from reviewer.citations.commands import report_dir as citations_dir
from reviewer.config import Settings
from reviewer.format.commands import report_dir as format_dir
from reviewer.mcp.config import McpConfig
from reviewer.mcp.format_tool import VenueId, local_pdf, run_check_format
from reviewer.mcp.index import HostedIndex
from reviewer.mcp.local_parse import is_local_url
from reviewer.mcp.prior_work import QueryError
from reviewer.mcp.review_summary import meta_review_out, summarize
from reviewer.models import ReviewPaperResult, ReviewStep, ReviewStepSummary, StepResult
from reviewer.parse.grobid_client import GrobidClient
from reviewer.parse.ids import paper_id_for
from reviewer.review.budget import Budget
from reviewer.review.estimate import estimate
from reviewer.review.orchestrator import VersionContext, run_review
from reviewer.review.profile import NOVELTY_MODES, init_work_dir, paper_settings
from reviewer.writing.commands import report_dir as writing_dir

logger = logging.getLogger(__name__)

NO_KEY = (
    "review_paper needs ANTHROPIC_API_KEY in the environment that launches the MCP server "
    "(it runs on your key and your bill). Without a key you can use check_format, "
    "check_citations, find_prior_work, paper and citations."
)
NO_GROBID = (
    "review_paper needs GROBID running on this machine to read the paper: "
    "docker run --rm -p 8070:8070 grobid/grobid:0.9.1-crf  (then call again). "
    "check_format, check_citations, find_prior_work, paper and citations work without it."
)
VENUE_META: dict[str, tuple[str, int | None]] = {
    "iclr2027": ("ICLR", 2027),
    "neurips2026": ("NeurIPS", 2026),
    "icml2026": ("ICML", 2026),
    "cvpr2027": ("CVPR", 2027),
    "acl": ("ACL", None),
    "other": ("Other", None),
}
STEPS = [
    ReviewStep.parse,
    ReviewStep.format,
    ReviewStep.citations,
    ReviewStep.claims,
    ReviewStep.novelty,
    ReviewStep.methods,
    ReviewStep.writing,
    ReviewStep.meta,
]


def _output_exists(step: ReviewStep, s: Settings, pid: str) -> bool:
    paths = {
        ReviewStep.parse: [s.parsed_dir / f"{pid}.json"],
        ReviewStep.claims: [s.claims_dir / f"{pid}.json"],
        ReviewStep.novelty: [s.novelty_dir / f"{pid}.json"],
        ReviewStep.methods: [s.methods_dir / f"{pid}.json"],
        ReviewStep.meta: [s.meta_dir / f"{pid}.json"],
        ReviewStep.format: [format_dir(s, p) / f"{pid}.json" for p in (True, False)],
        ReviewStep.citations: [citations_dir(s, p) / f"{pid}.json" for p in (True, False)],
        ReviewStep.writing: [writing_dir(s, p) / f"{pid}.json" for p in (True, False)],
    }[step]
    return any(p.exists() for p in paths)


def _settings(config: McpConfig, sha: str) -> Settings:
    base = config.pipeline_settings()
    paper_dir = config.cache / "papers" / sha
    return paper_settings(base, paper_dir / "work", config.cache / "public", paper_dir)


def _gate(s: Settings) -> None:
    if not s.anthropic_api_key:
        raise QueryError(NO_KEY)
    if not is_local_url(s.grobid_url):
        raise QueryError(NO_GROBID + " (GROBID_URL must point at this machine.)")
    client = GrobidClient(s)
    try:
        if not client.is_alive():
            raise QueryError(NO_GROBID)
    finally:
        client.close()


NoveltyMode = Literal["lite", "full"]
MODE_FILE = "novelty_mode.txt"
"""Which novelty mode produced the novelty (and so meta) output in this work dir."""


def run_review_paper(
    pdf_path: str,
    venue: VenueId,
    confirm: bool = False,
    config: McpConfig | None = None,
    on_step: Callable[[StepResult], None] | None = None,
    novelty: NoveltyMode = "full",
) -> ReviewPaperResult:
    config = config or McpConfig()
    pdf = local_pdf(pdf_path)
    if venue not in VENUE_META:
        raise QueryError("venue must be one of " + ", ".join(VENUE_META) + ".")
    if novelty not in NOVELTY_MODES:
        raise QueryError("novelty must be 'lite' or 'full'.")
    sha = sha256_file(pdf)
    s = _settings(config, sha).model_copy(update=NOVELTY_MODES[novelty])
    _gate(s)
    init_work_dir(s)
    pid = paper_id_for(pdf, sha)
    fmt = run_check_format(str(pdf), venue, "submission", config)
    mode_file = s.data_dir / MODE_FILE
    previous_mode = mode_file.read_text().strip() if mode_file.exists() else None
    stale = {ReviewStep.novelty, ReviewStep.meta} if previous_mode not in (None, novelty) else set()
    done = [st for st in STEPS if st not in stale and _output_exists(st, s, pid)]
    todo = [st for st in STEPS if st not in done]
    about, _reserve, limit = estimate(todo, novelty)
    common = {
        "estimate_usd": about,
        "limit_usd": limit,
        "steps_to_run": [st.value for st in todo],
        "steps_already_done": [st.value for st in done],
        "novelty_mode": novelty,
        "format": fmt,
    }
    if not confirm:
        return ReviewPaperResult(
            phase="plan",
            notes=[
                f"About ${about:.2f}, never more than ${limit:.2f}, on your Anthropic key. Ask "
                "the user, then call review_paper again with confirm=true."
            ],
            **common,
        )
    # A review makes 100+ index requests in minutes: pace into the per-IP limit instead of
    # failing (a chat tool keeps the short 5 s wait).
    index = HostedIndex(
        config.model_copy(update={"index_retries": 6, "index_max_retry_after_s": 30.0})
    )
    index.paper("1706.03762")  # surface a missing/bad index key before spending anything
    name, year = VENUE_META[venue]
    ctx = VersionContext(
        paper_id=pid, settings=s, venue=name, venue_year=year, pdf=pdf, index=index
    )
    budget = Budget(limit_usd=max(limit, 0.01))
    results = run_review(ctx, budget, on_step)
    if any(r.step == ReviewStep.novelty and r.status.value == "done" for r in results):
        mode_file.write_text(novelty + "\n")
    notes: list[str] = []
    meta = meta_review_out(results, s, name, year, ctx.failures)
    if meta is None:
        notes.append("No meta-review: its step did not finish (see steps).")
    elif meta.left_out_note:
        notes.append(meta.left_out_note)
    unfinished = [r for r in results if r.status.value != "done"]
    if unfinished:
        notes.append(
            "Not finished: "
            + ", ".join(f"{r.step.value} ({r.error_kind})" for r in unfinished)
            + ". Call review_paper again with the same PDF to retry; finished steps are free."
        )
    return ReviewPaperResult(
        phase="review",
        spent_usd=round(budget.spent_usd, 4),
        steps=[
            ReviewStepSummary(
                step=r.step.value,
                status=r.status.value,
                cost_usd=r.cost_usd,
                error_kind=r.error_kind,
            )
            for r in results
        ],
        findings=summarize(results, Path(s.data_dir)),
        meta_review=meta,
        notes=notes,
        **common,
    )
