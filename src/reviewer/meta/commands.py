"""Orchestration: make sure each checker has run for a paper, assemble MetaInput, judge,
calibrate, render. `reviewer meta review` (one paper) and `reviewer meta batch` (many, with a
spend guard -- plan §10's $120 hard cap)."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from reviewer.citations.commands import report_dir as citations_report_dir
from reviewer.citations.commands import run_citations
from reviewer.claims.commands import extract_one
from reviewer.config import PROJECT_ROOT, Settings, get_settings
from reviewer.corpus.corpus import load_corpus
from reviewer.failures import record_failure
from reviewer.format.commands import report_dir as format_report_dir
from reviewer.format.commands import run_format
from reviewer.llm.client import LLMClient, MissingAPIKeyError
from reviewer.meta.baselines import (
    BaselineJudgeError,
    run_baseline_abstract,
    run_baseline_full,
    run_contamination_probe,
)
from reviewer.meta.blinding import blind_paper
from reviewer.meta.calibration import calibrate
from reviewer.meta.input_builder import build_meta_input
from reviewer.meta.judge import MetaJudgeError, checker_ok_count, judge_paper_k
from reviewer.meta.prompts import load_meta_prompt
from reviewer.meta.report import render_review
from reviewer.methods.commands import run_check as methods_run_check
from reviewer.models import (
    CorpusEntry,
    MetaReview,
    PaperClaims,
    PaperTier,
    ParsedPaper,
    TokenUsage,
)
from reviewer.novelty.commands import run_check as novelty_run_check
from reviewer.privacy import write_private
from reviewer.writing.commands import report_dir as writing_report_dir
from reviewer.writing.commands import run_writing

logger = logging.getLogger(__name__)


class BudgetExceededError(RuntimeError):
    pass


class SpendGuard:
    """Tracks fresh spend across checkers (read post-hoc from each output's own token_usage,
    since not every checker command takes an injected client) and the meta LLM calls."""

    def __init__(self, cap_usd: float) -> None:
        self.cap_usd = cap_usd
        self.spent_usd = 0.0

    def add(self, usage: TokenUsage) -> None:
        self.spent_usd += usage.cost_usd

    def check(self, paper_id: str) -> None:
        if self.spent_usd > self.cap_usd:
            raise BudgetExceededError(
                f"spend ${self.spent_usd:.2f} exceeds the ${self.cap_usd:.2f} cap "
                f"(hit while processing {paper_id})"
            )


def _cost_of_json(path: Path, usage_path: tuple[str, ...]) -> float:
    """Read `token_usage.cost_usd`-shaped fields out of a checker's own JSON output, tolerating
    a missing file or field (e.g. the checker was skipped)."""
    if not path.exists():
        return 0.0
    obj: object = json.loads(path.read_text(encoding="utf-8"))
    for key in usage_path:
        if not isinstance(obj, dict) or key not in obj:
            return 0.0
        obj = obj[key]
    return float(obj) if isinstance(obj, (int, float)) else 0.0


def _top_claims_only(claims: PaperClaims, n: int) -> PaperClaims:
    kept = sorted(claims.claims, key=lambda c: -c.confidence)[:n]
    return claims.model_copy(update={"claims": kept})


def ensure_checkers(
    paper: ParsedPaper,
    settings: Settings,
    guard: SpendGuard,
    *,
    lite: bool,
    force: bool = False,
) -> dict[str, str]:
    """Run any checker that hasn't produced output for this paper yet. Returns {checker: error}
    for ones that failed; missing ones are left for MetaInput to report as `not_run`."""
    pid = paper.paper_id
    failures: dict[str, str] = {}

    claims_path = settings.claims_dir / f"{pid}.json"
    if force or not claims_path.exists():
        try:
            client = LLMClient(settings)
            ok = extract_one(
                settings.parsed_dir / f"{pid}.json",
                settings.llm_model,
                settings.claims_dir,
                force,
                client,
                settings,
            )
            guard.add(client.totals.fresh)
            if not ok:
                failures["claims"] = "extraction failed (see data/failures.jsonl)"
        except MissingAPIKeyError:
            raise
    guard.check(pid)

    novelty_path = settings.novelty_dir / f"{pid}.json"
    claims = (
        PaperClaims.model_validate_json(claims_path.read_text(encoding="utf-8"))
        if claims_path.exists()
        else None
    )
    if claims is not None and not claims.claims:
        logger.info("%s: 0 claims, so novelty has nothing to check; skipped", pid)
    elif (force or not novelty_path.exists()) and claims is not None:
        claims_dir = settings.claims_dir
        if lite and settings.meta_novelty_top_claims:
            lite_dir = settings.data_dir / "claims_meta_lite"
            lite_dir.mkdir(parents=True, exist_ok=True)
            (lite_dir / f"{pid}.json").write_text(
                _top_claims_only(claims, settings.meta_novelty_top_claims).model_dump_json()
            )
            claims_dir = lite_dir
        try:
            novelty_run_check(
                ids=[pid],
                force=force,
                claims_dir=claims_dir,
                settings=settings,
                extra_entries=[
                    CorpusEntry(paper_id=pid, tier=PaperTier.ordinary, reason="meta dev run")
                ],
            )
        except Exception as e:
            failures["novelty"] = f"{type(e).__name__}: {e}"
            record_failure(settings.failures_file, "novelty", pid, failures["novelty"])
        guard.add(TokenUsage(cost_usd=_cost_of_json(novelty_path, ("token_usage", "cost_usd"))))
    guard.check(pid)

    for name, out_dir_fn, run_fn, kwargs in (
        ("citations", citations_report_dir, run_citations, {}),
        ("format", format_report_dir, run_format, {"pdf_root": PROJECT_ROOT}),
        ("writing", writing_report_dir, run_writing, {"force": force}),
    ):
        out_private = out_dir_fn(settings, True) / f"{pid}.json"
        out_public = out_dir_fn(settings, False) / f"{pid}.json"
        if force or not (out_private.exists() or out_public.exists()):
            try:
                run_fn(settings.parsed_dir / f"{pid}.json", settings=settings, **kwargs)
            except Exception as e:
                failures[name] = f"{type(e).__name__}: {e}"
                record_failure(settings.failures_file, name, pid, failures[name])
            cost = max(
                _cost_of_json(out_private, ("token_usage", "cost_usd")),
                _cost_of_json(out_public, ("token_usage", "cost_usd")),
            )
            guard.add(TokenUsage(cost_usd=cost))
        guard.check(pid)

    methods_path = settings.methods_dir / f"{pid}.json"
    if force or not methods_path.exists():
        try:
            methods_run_check(ids=[pid], force=force, settings=settings)
        except Exception as e:
            failures["methods"] = f"{type(e).__name__}: {e}"
            record_failure(settings.failures_file, "methods", pid, failures["methods"])
        guard.add(TokenUsage(cost_usd=_cost_of_json(methods_path, ("token_usage", "cost_usd"))))
    guard.check(pid)

    return failures


def run_meta_review(
    paper: ParsedPaper,
    *,
    venue: str,
    venue_year: int | None,
    settings: Settings,
    client: LLMClient,
    failures: dict[str, str] | None = None,
    k: int | None = None,
) -> MetaReview:
    mi = build_meta_input(settings, paper, venue=venue, venue_year=venue_year, failures=failures)
    prompt = load_meta_prompt(settings.meta_prompt)
    judgments, dropped, usage = judge_paper_k(
        client,
        mi,
        prompt=prompt,
        model=settings.meta_model,
        max_retries=settings.llm_max_retries,
        k=k if k is not None else settings.meta_k,
    )
    cal = calibrate(judgments)

    baseline_abstract = baseline_full = contamination_probe = None
    if settings.meta_run_baselines:
        for label, call in (
            (
                "baseline_abstract",
                lambda: run_baseline_abstract(
                    client,
                    blind_paper(paper),
                    prompt=load_meta_prompt(settings.baseline_abstract_prompt),
                    model=settings.meta_model,
                    venue=venue,
                    venue_year=venue_year,
                    max_retries=settings.llm_max_retries,
                ),
            ),
            (
                "baseline_full",
                lambda: run_baseline_full(
                    client,
                    mi.blinded_text,
                    prompt=load_meta_prompt(settings.baseline_full_prompt),
                    model=settings.meta_model,
                    venue=venue,
                    venue_year=venue_year,
                    max_retries=settings.llm_max_retries,
                ),
            ),
            (
                "contamination_probe",
                lambda: run_contamination_probe(
                    client,
                    blind_paper(paper).title,
                    prompt=load_meta_prompt(settings.contamination_prompt),
                    model=settings.meta_model,
                    venue=venue,
                    venue_year=venue_year,
                    max_retries=settings.llm_max_retries,
                ),
            ),
        ):
            try:
                out, call_usage = call()
                usage += call_usage
            except BaselineJudgeError as e:
                logger.warning("meta %s failed for %s: %s", label, paper.paper_id, e)
                usage += e.usage or TokenUsage()
                continue
            if label == "baseline_abstract":
                baseline_abstract = out
            elif label == "baseline_full":
                baseline_full = out
            else:
                contamination_probe = out

    review = MetaReview(
        paper_id=paper.paper_id,
        title=paper.title,
        venue=venue,
        venue_year=venue_year,
        model=settings.meta_model,
        prompt_version=prompt.version,
        generated=datetime.now(UTC),
        judgments=judgments,
        dropped_evidence_ids=dropped,
        calibration=cal,
        checker_status=mi.checker_status,
        token_usage=usage,
        baseline_abstract=baseline_abstract,
        baseline_full=baseline_full,
        contamination_probe=contamination_probe,
    )
    logger.info(
        "meta review %s: %d/%d checkers ok, raw rating(s) %s, %s",
        paper.paper_id,
        checker_ok_count(mi),
        len(mi.checker_status),
        [j.raw_rating for j in judgments],
        f"${usage.cost_usd:.2f}",
    )
    return review


def write_meta_review(settings: Settings, review: MetaReview) -> Path:
    out = settings.meta_dir / f"{review.paper_id}.json"
    write_private(out, review.model_dump_json(indent=2) + "\n")
    write_private(out.with_suffix(".md"), render_review(review))
    return out


def run_batch(
    paper_ids: list[str],
    *,
    venue: str = "ICLR",
    venue_year: int | None = None,
    lite: bool = True,
    force: bool = False,
    settings: Settings | None = None,
    venue_years: dict[str, int] | None = None,
) -> list[MetaReview]:
    """`venue_years`: per-paper venue year (overrides `venue_year`), for batches spanning
    several years of one venue."""
    settings = settings or get_settings()
    guard = SpendGuard(settings.meta_budget_usd)
    tiers = {e.paper_id: e.tier.value for e in load_corpus(settings)}
    client = LLMClient(settings)
    reviews = []
    for pid in paper_ids:
        parsed_path = settings.parsed_dir / f"{pid}.json"
        if not parsed_path.exists():
            logger.warning("meta batch: no parsed paper for %s, skipping", pid)
            continue
        paper = ParsedPaper.model_validate_json(parsed_path.read_text(encoding="utf-8"))
        fresh_before = client.totals.fresh.cost_usd
        try:
            failures = ensure_checkers(paper, settings, guard, lite=lite, force=force)
            review = run_meta_review(
                paper,
                venue=venue,
                venue_year=(venue_years or {}).get(pid, venue_year),
                settings=settings,
                client=client,
                failures=failures,
            )
        except BudgetExceededError:
            logger.error(
                "meta batch: budget exceeded, stopping (%d papers left)",
                len(paper_ids) - len(reviews),
            )
            raise
        except MetaJudgeError as e:
            # cached responses cost nothing now: count only this run's fresh spend
            guard.add(TokenUsage(cost_usd=client.totals.fresh.cost_usd - fresh_before))
            record_failure(settings.failures_file, "meta", pid, str(e))
            continue
        guard.add(TokenUsage(cost_usd=client.totals.fresh.cost_usd - fresh_before))
        write_meta_review(settings, review)
        reviews.append(review)
        logger.info(
            "meta batch: %s tier=%s done, spend so far $%.2f/$%.2f",
            pid,
            tiers.get(pid, "?"),
            guard.spent_usd,
            settings.meta_budget_usd,
        )
    return reviews
