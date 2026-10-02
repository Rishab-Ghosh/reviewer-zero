"""`run_step(step, ctx, budget)`: one review step over the existing checkers (BP §4).

Each step runs with a copy of the paper's settings carrying the step's `max_tokens` and a call
guard that takes a `Budget` hold before every uncached LLM call and refuses new calls after the
step's deadline. Checkers read and write their usual paths under the paper's work dir.
"""

import json
import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import typer

from reviewer.citations.commands import report_dir as citations_report_dir
from reviewer.citations.commands import run_citations
from reviewer.claims.commands import extract_one
from reviewer.config import PROJECT_ROOT, Settings
from reviewer.format.commands import report_dir as format_report_dir
from reviewer.format.commands import run_format
from reviewer.llm.client import LLMClient, TransportRequest
from reviewer.meta.calibration import calibrate
from reviewer.meta.input_builder import build_meta_input
from reviewer.meta.judge import judge_paper_k
from reviewer.meta.prompts import load_meta_prompt
from reviewer.methods.commands import run_check as methods_run_check
from reviewer.models import (
    CorpusEntry,
    MetaReview,
    PaperClaims,
    PaperTier,
    ParsedPaper,
    ReviewStep,
    StepResult,
    StepStatus,
)
from reviewer.novelty.commands import run_check as novelty_run_check
from reviewer.novelty.index_client import IndexAPI
from reviewer.parse.commands import run_parse
from reviewer.privacy import write_private
from reviewer.review.budget import (
    Budget,
    BudgetExceeded,
    Slot,
    StepTimeout,
    estimate_input_tokens,
    hold_usd,
)
from reviewer.review.caps import load_caps
from reviewer.review.profile import init_work_dir
from reviewer.writing.commands import report_dir as writing_report_dir
from reviewer.writing.commands import run_writing

logger = logging.getLogger(__name__)

STEP_MAX_TOKENS: dict[ReviewStep, int] = load_caps()
"""2x the largest observed output of each step's active prompts (task 1.8, `review/caps.py`)."""


@dataclass
class VersionContext:
    """One paper version being reviewed."""

    paper_id: str
    settings: Settings
    """Per-paper settings (see `review.profile.paper_settings`)."""
    venue: str = "ICLR"
    venue_year: int | None = None
    pdf: Path | None = None
    failures: dict[str, str] = field(default_factory=dict)
    """{checker: error kind} of earlier steps, passed to the meta step."""
    index: IndexAPI | None = None
    """Index client for the novelty step; None = the pipeline's default `HttpIndexClient`
    (the MCP server passes its `HostedIndex`: API key, batched /expand)."""

    @property
    def parsed_path(self) -> Path:
        return self.settings.parsed_dir / f"{self.paper_id}.json"

    def paper(self) -> ParsedPaper:
        return ParsedPaper.model_validate_json(self.parsed_path.read_text(encoding="utf-8"))


class StepFailedError(RuntimeError):
    """A step failure with a fixed, text-free kind (safe to log and store)."""

    def __init__(self, kind: str) -> None:
        super().__init__(kind)
        self.kind = kind


@dataclass
class _GuardState:
    costs: list[float] = field(default_factory=list)
    stopped_by: str | None = None
    """'timeout' or 'budget' once the guard refused a call (checkers may swallow the error)."""
    lock: threading.Lock = field(default_factory=threading.Lock)


def _schema_text(req: TransportRequest) -> str:
    return json.dumps(req.schema.model_json_schema())


def make_call_guard(
    budget: Budget, deadline: float | None, state: _GuardState
) -> Callable[[TransportRequest], object]:
    @contextmanager
    def guard(req: TransportRequest) -> Iterator[Slot]:
        tokens = estimate_input_tokens(req.system, req.user, _schema_text(req))
        hold = hold_usd(req.model, tokens, req.max_tokens)
        slot: Slot | None = None
        try:
            with budget.call(hold, deadline) as slot:
                yield slot
        except (StepTimeout, BudgetExceeded) as e:
            if slot is None:  # refused at admission; the checker may swallow the error
                with state.lock:
                    state.stopped_by = "timeout" if isinstance(e, StepTimeout) else "budget"
            raise
        finally:
            if slot is not None:
                with state.lock:
                    state.costs.append(slot.actual_usd or 0.0)

    return guard


# --------------------------------------------------------------------------- steps


def _existing(*paths: Path) -> Path | None:
    return next((p for p in paths if p.exists()), None)


def _report_output(dir_fn: Callable[[Settings, bool], Path], s: Settings, pid: str) -> Path | None:
    return _existing(dir_fn(s, True) / f"{pid}.json", dir_fn(s, False) / f"{pid}.json")


def _step_parse(ctx: VersionContext, s: Settings) -> Path | None:
    if ctx.pdf is None:
        raise ValueError("parse needs ctx.pdf")
    try:
        run_parse(ctx.pdf, settings=s)
    except typer.Exit as e:
        raise StepFailedError("grobid_unavailable") from e
    return _existing(ctx.parsed_path)


def _step_format(ctx: VersionContext, s: Settings) -> Path | None:
    run_format(ctx.parsed_path, pdf_root=PROJECT_ROOT, settings=s)
    return _report_output(format_report_dir, s, ctx.paper_id)


def _step_citations(ctx: VersionContext, s: Settings) -> Path | None:
    run_citations(ctx.parsed_path, settings=s)
    return _report_output(citations_report_dir, s, ctx.paper_id)


def _step_claims(ctx: VersionContext, s: Settings) -> Path | None:
    ok = extract_one(ctx.parsed_path, s.llm_model, s.claims_dir, False, LLMClient(s), s)
    return s.claims_dir / f"{ctx.paper_id}.json" if ok else None


def _step_novelty(ctx: VersionContext, s: Settings) -> Path | None:
    claims_path = s.claims_dir / f"{ctx.paper_id}.json"
    claims = PaperClaims.model_validate_json(claims_path.read_text(encoding="utf-8"))
    if not claims.claims:
        return None
    claims_dir = s.claims_dir
    if s.meta_novelty_top_claims:
        # Same top-N-by-confidence selection as `ensure_checkers(lite=True)`.
        kept = sorted(claims.claims, key=lambda c: -c.confidence)[: s.meta_novelty_top_claims]
        claims_dir = s.data_dir / "claims_meta_lite"
        claims_dir.mkdir(parents=True, exist_ok=True)
        (claims_dir / f"{ctx.paper_id}.json").write_text(
            claims.model_copy(update={"claims": kept}).model_dump_json()
        )
    novelty_run_check(
        index=ctx.index,
        ids=[ctx.paper_id],
        claims_dir=claims_dir,
        settings=s,
        extra_entries=[
            CorpusEntry(paper_id=ctx.paper_id, tier=PaperTier.ordinary, reason="web review")
        ],
    )
    return _existing(s.novelty_dir / f"{ctx.paper_id}.json")


def _step_methods(ctx: VersionContext, s: Settings) -> Path | None:
    methods_run_check(ids=[ctx.paper_id], settings=s)
    return _existing(s.methods_dir / f"{ctx.paper_id}.json")


def _step_writing(ctx: VersionContext, s: Settings) -> Path | None:
    run_writing(ctx.parsed_path, settings=s)
    return _report_output(writing_report_dir, s, ctx.paper_id)


def _step_meta(ctx: VersionContext, s: Settings) -> Path | None:
    """The judgment path only (D67): no baselines, no contamination probe, `meta_k` samples."""
    paper = ctx.paper()
    mi = build_meta_input(
        s, paper, venue=ctx.venue, venue_year=ctx.venue_year, failures=ctx.failures
    )
    prompt = load_meta_prompt(s.meta_prompt)
    judgments, dropped, usage = judge_paper_k(
        LLMClient(s),
        mi,
        prompt=prompt,
        model=s.meta_model,
        max_retries=s.llm_max_retries,
        k=s.meta_k,
    )
    review = MetaReview(
        paper_id=paper.paper_id,
        title=paper.title,
        venue=ctx.venue,
        venue_year=ctx.venue_year,
        model=s.meta_model,
        prompt_version=prompt.version,
        generated=datetime.now(UTC),
        judgments=judgments,
        dropped_evidence_ids=dropped,
        calibration=calibrate(judgments),
        checker_status=mi.checker_status,
        token_usage=usage,
    )
    out = s.meta_dir / f"{paper.paper_id}.json"
    write_private(out, review.model_dump_json(indent=2) + "\n")
    return out


STEPS: dict[ReviewStep, Callable[[VersionContext, Settings], Path | None]] = {
    ReviewStep.parse: _step_parse,
    ReviewStep.format: _step_format,
    ReviewStep.citations: _step_citations,
    ReviewStep.claims: _step_claims,
    ReviewStep.novelty: _step_novelty,
    ReviewStep.methods: _step_methods,
    ReviewStep.writing: _step_writing,
    ReviewStep.meta: _step_meta,
}


def run_step(
    step: ReviewStep,
    ctx: VersionContext,
    budget: Budget,
    deadline: float | None = None,
) -> StepResult:
    """Run one step. `deadline` is a `time.monotonic()` instant after which no new LLM call
    starts. Never raises for a checker failure: the result says what happened."""
    state = _GuardState()
    s = ctx.settings.model_copy(
        update={"llm_max_tokens": STEP_MAX_TOKENS[step] or 1}
    ).with_call_guard(make_call_guard(budget, deadline, state))
    init_work_dir(s)
    t0 = time.monotonic()
    output: Path | None = None
    error: str | None = None
    try:
        output = STEPS[step](ctx, s)
    except StepTimeout:
        error = "timeout"
    except BudgetExceeded:
        error = "budget"
    except StepFailedError as e:
        error = e.kind
    except Exception as e:
        cause = e.__cause__
        # Type names only (never messages: they can quote the paper).
        error = type(e).__name__ + (f"({type(cause).__name__})" if cause is not None else "")
        logger.warning("step %s on %s failed: %s", step, ctx.paper_id, type(e).__name__)
    if state.stopped_by is not None:
        error = state.stopped_by
    if error == "timeout":
        status = StepStatus.timed_out
    elif error is not None or output is None:
        status = StepStatus.failed
        error = error or "no_output"
    else:
        status = StepStatus.done
    if status != StepStatus.done:
        ctx.failures[step.value] = error or "failed"
    result = StepResult(
        step=step,
        status=status,
        cost_usd=round(sum(state.costs), 6),
        n_calls=len(state.costs),
        seconds=round(time.monotonic() - t0, 2),
        error_kind=error,
        output=str(output) if output and status == StepStatus.done else None,
    )
    logger.info(
        "step %s on %s: %s, $%.4f over %d calls, %.1fs",
        step,
        ctx.paper_id,
        result.status,
        result.cost_usd,
        result.n_calls,
        result.seconds,
    )
    return result


# --------------------------------------------------------------------------- job

CHECKERS_AFTER_PARSE = (
    ReviewStep.format,
    ReviewStep.citations,
    ReviewStep.claims,
    ReviewStep.methods,
    ReviewStep.writing,
)
META_MIN_OTHER_OK = 3
"""Meta runs only if claims succeeded and at least 3 of the 5 other checkers did (BP §4)."""


def run_review(
    ctx: VersionContext,
    budget: Budget,
    on_step: Callable[[StepResult], None] | None = None,
) -> list[StepResult]:
    """The W1 DAG (BP §4): parse, then format/citations/claims/methods/writing in parallel,
    novelty after claims, meta last. No findings or anchors yet (W2-W3). `on_step` is called
    (from worker threads) as each step finishes."""
    from concurrent.futures import ThreadPoolExecutor

    def step(s: ReviewStep) -> StepResult:
        r = run_step(s, ctx, budget)
        if on_step is not None:
            on_step(r)
        return r

    results: dict[ReviewStep, StepResult] = {}
    results[ReviewStep.parse] = step(ReviewStep.parse)
    if results[ReviewStep.parse].status != StepStatus.done:
        return list(results.values())

    def claims_then_novelty() -> list[StepResult]:
        claims = step(ReviewStep.claims)
        if claims.status != StepStatus.done:
            skipped = StepResult(
                step=ReviewStep.novelty, status=StepStatus.skipped, error_kind="claims_failed"
            )
            return [claims, skipped]
        return [claims, step(ReviewStep.novelty)]

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(claims_then_novelty)] + [
            pool.submit(lambda s=s: [step(s)])
            for s in CHECKERS_AFTER_PARSE
            if s != ReviewStep.claims
        ]
        for f in futures:
            for r in f.result():
                results[r.step] = r

    others = [ReviewStep.novelty, ReviewStep.citations, ReviewStep.methods, ReviewStep.format]
    others.append(ReviewStep.writing)
    n_ok = sum(results[s].status == StepStatus.done for s in others)
    if results[ReviewStep.claims].status == StepStatus.done and n_ok >= META_MIN_OTHER_OK:
        results[ReviewStep.meta] = step(ReviewStep.meta)
    else:
        results[ReviewStep.meta] = StepResult(
            step=ReviewStep.meta, status=StepStatus.skipped, error_kind="too_few_checkers"
        )
    order = [ReviewStep.parse, *CHECKERS_AFTER_PARSE, ReviewStep.novelty, ReviewStep.meta]
    return [results[s] for s in order if s in results]
