"""`reviewer novelty check` and `reviewer novelty eval`. All outputs are private
(reports/private/): lab-tier and own-paper results must not be committed."""

import json
import logging
import math
import subprocess
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import typer

from reviewer.config import PROJECT_ROOT, Settings, get_settings
from reviewer.corpus.corpus import load_corpus
from reviewer.failures import record_failure
from reviewer.llm.client import (
    LLMAPIError,
    LLMClient,
    LLMError,
    MissingAPIKeyError,
    TransportRequest,
    TransportResponse,
)
from reviewer.models import (
    ClaimNovelty,
    CorpusEntry,
    NegativeCaseRow,
    NoveltyEvalCase,
    NoveltyEvalCaseResult,
    NoveltyVerdict,
    PaperClaims,
    PaperNovelty,
    PaperTier,
    ParsedPaper,
)
from reviewer.novelty.checker import Checker
from reviewer.novelty.evaluate import (
    auc,
    by_query_group,
    case_query,
    case_target,
    label_counts,
    load_cases,
    matches,
    micro,
    sample_cases,
    score_case,
    score_retrieval,
)
from reviewer.novelty.firststage import (
    CrossValidatedScorer,
    StageCandidate,
    cross_validated_scores,
    feature_matrix,
    fit_folds,
    fit_logistic,
    gather_candidates,
    rule_score,
    select_pool,
)
from reviewer.novelty.index_client import HttpIndexClient, IndexAPI, IndexClientError
from reviewer.novelty.queries import QueryGenerationError
from reviewer.novelty.report import (
    render_detail,
    render_eval,
    render_first_stage,
    render_negatives,
    render_report,
    render_severity,
)
from reviewer.novelty.rerank import RerankError, load_rerank_prompt
from reviewer.novelty.retrieve import (
    baseline_queries,
    resolve_v1_date,
    retrieve_multi,
    target_for,
)
from reviewer.privacy import append_private, write_private

logger = logging.getLogger(__name__)

DEFAULT_TIERS = (PaperTier.lab, PaperTier.own)
REPORT_NAME = "novelty_v0.md"
EVAL_NAME = "novelty_eval"  # + _dev / _test


def select_entries(
    entries: list[CorpusEntry],
    ids: list[str] | None,
    tiers: list[PaperTier] | None,
    extra: list[CorpusEntry] | None = None,
) -> list[CorpusEntry]:
    """`extra`: entries for ids that are not, and should not be, in corpus.yaml (e.g. the
    meta-reviewer's OpenReview dev-run papers) -- looked up only when `ids` names them."""
    if ids:
        by_id = {e.paper_id: e for e in (extra or [])}
        by_id.update({e.paper_id: e for e in entries})
        unknown = [i for i in ids if i not in by_id]
        if unknown:
            raise typer.BadParameter(f"not in corpus: {unknown}")
        return [by_id[i] for i in ids]
    wanted = set(tiers or DEFAULT_TIERS)
    return [e for e in entries if e.tier in wanted]


def _check_one(
    entry: CorpusEntry,
    checker: Checker,
    settings: Settings,
    force: bool,
    claims_dir: Path | None = None,
) -> PaperNovelty | None:
    out = settings.novelty_dir / f"{entry.paper_id}.json"
    if out.exists() and not force:
        logger.info("skip %s: %s exists (use --force)", entry.paper_id, out)
        return PaperNovelty.model_validate_json(out.read_text(encoding="utf-8"))
    claims_path = (claims_dir or settings.claims_dir) / f"{entry.paper_id}.json"
    if not claims_path.exists():
        record_failure(
            settings.failures_file, "novelty", entry.paper_id, f"no claims file {claims_path}"
        )
        return None
    claims = PaperClaims.model_validate_json(claims_path.read_text(encoding="utf-8"))
    abstract = paper_abstract(settings, claims)
    target = target_for(entry.paper_id, claims.title, cutoff=True, abstract=abstract)
    try:
        target = resolve_v1_date(checker.index, target)
        result = checker.check_paper(claims, entry.tier.value, target, abstract)
    except MissingAPIKeyError:
        raise
    except (IndexClientError, LLMError, RerankError, QueryGenerationError) as e:
        record_failure(
            settings.failures_file, "novelty", entry.paper_id, f"{type(e).__name__}: {e}"
        )
        return None
    write_private(out, result.model_dump_json(indent=2) + "\n")
    return result


def paper_abstract(settings: Settings, claims: PaperClaims) -> str:
    """The parsed paper's abstract (arXiv's for arXiv papers), else the one-line summary."""
    parsed = settings.parsed_dir / f"{claims.paper_id}.json"
    if parsed.exists():
        abstract = ParsedPaper.model_validate_json(parsed.read_text(encoding="utf-8")).abstract
        if abstract:
            return abstract
    logger.warning("%s: no parsed abstract; using the one-line summary", claims.paper_id)
    return claims.one_line_summary


def _checker(settings: Settings, index: IndexAPI | None, client: LLMClient | None) -> Checker:
    index = index or HttpIndexClient(settings)
    try:
        index.version()
    except IndexClientError as e:
        logger.error("reviewer-index unavailable at %s: %s", settings.index_url, e)
        raise typer.Exit(code=1) from e
    return Checker.from_settings(
        settings,
        index,
        client or LLMClient(settings),
        load_rerank_prompt(settings.novelty_prompt),
        settings.llm_model,
    )


def run_check(
    ids: list[str] | None = None,
    tiers: list[PaperTier] | None = None,
    force: bool = False,
    detail: bool = False,
    claims_dir: Path | None = None,
    settings: Settings | None = None,
    index: IndexAPI | None = None,
    client: LLMClient | None = None,
    extra_entries: list[CorpusEntry] | None = None,
) -> None:
    """Check every claim of the selected papers (default: lab + own tiers). `extra_entries`:
    see `select_entries`."""
    settings = settings or get_settings()
    entries = select_entries(load_corpus(settings), ids, tiers, extra_entries)
    checker = _checker(settings, index, client)
    logger.info("novelty check: %d papers", len(entries))
    results: list[PaperNovelty] = []
    failed: list[str] = []
    try:
        for entry in entries:
            result = _check_one(entry, checker, settings, force, claims_dir)
            if result is None:
                failed.append(entry.paper_id)
                continue
            results.append(result)
            if detail:
                write_private(
                    settings.private_reports_dir / f"novelty_detail_{entry.paper_id}.md",
                    render_detail(result),
                )
    except MissingAPIKeyError as e:
        logger.error("%s", e)
        raise typer.Exit(code=1) from e
    failures = [f"{pid}: see data/failures.jsonl (stage novelty)" for pid in failed]
    write_private(settings.private_reports_dir / REPORT_NAME, render_report(results, failures))
    fresh = checker.client.totals.fresh
    logger.info(
        "novelty check: %d/%d papers ok; new LLM spend $%.4f (%d calls, %d cached)",
        len(results),
        len(entries),
        fresh.cost_usd,
        checker.client.totals.n_fresh,
        checker.client.totals.n_cached,
    )


Split = Literal["dev", "test"]
EVAL_RUNS_LOG = "novelty_eval_runs.jsonl"


def git_state() -> tuple[str, bool]:
    """(HEAD commit, whether tracked files have uncommitted changes)."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    return git("rev-parse", "HEAD"), bool(git("status", "--porcelain", "--untracked-files=no"))


def eval_cases_path(settings: Settings, split: Split, cases_path: Path | None) -> Path:
    if cases_path is None:
        return settings.novelty_eval_dir / f"{split}.jsonl"
    if split == "dev" and cases_path.name.startswith("test"):
        raise typer.BadParameter(f"{cases_path} looks like the test split; use --split test")
    return cases_path


def check_test_allowed(split: Split, confirm_test: bool) -> tuple[str, bool]:
    """Test runs need explicit confirmation and a clean tree, so the logged commit hash is
    exactly the code that produced the numbers."""
    commit, dirty = git_state()
    if split == "test":
        if not confirm_test:
            logger.error("the test split runs only when asked for: pass --confirm-test")
            raise typer.Exit(code=1)
        if dirty:
            logger.error("test runs need a clean working tree (commit first); commit %s", commit)
            raise typer.Exit(code=1)
    return commit, dirty


def run_eval(
    split: Split = "dev",
    limit: int | None = None,
    confirm_test: bool = False,
    cases_path: Path | None = None,
    settings: Settings | None = None,
    index: IndexAPI | None = None,
    client: LLMClient | None = None,
    retrieval_only: bool = False,
) -> dict[str, float]:
    """Run the checker on each eval case and report recall@5/@20 of reviewer-named prior work.
    Every run is appended to reports/private/novelty_eval_runs.jsonl with its commit hash.
    `retrieval_only`: stop after the first-pass pool (union and pool recall only) and allow
    no new LLM calls (generated queries must already be cached)."""
    settings = settings or get_settings()
    cases_path = eval_cases_path(settings, split, cases_path)
    if not cases_path.exists():
        logger.error("no eval cases at %s (built by reviewer-index); nothing to do", cases_path)
        raise typer.Exit(code=1)
    commit, dirty = check_test_allowed(split, confirm_test)
    cases = sample_cases(load_cases(cases_path), limit)
    if retrieval_only:
        client = LLMClient(settings, transport=_no_new_llm_calls)
    checker = _checker(settings, index, client)
    logger.info("novelty eval on %s split: %d cases (commit %s)", split, len(cases), commit[:10])
    results: list[NoveltyEvalCaseResult] = []
    try:
        for case in cases:
            results.append(_eval_case(case, checker, settings, retrieval_only))
    except (MissingAPIKeyError, LLMCacheMissError) as e:
        logger.error("%s", e)
        raise typer.Exit(code=1) from e
    summary = micro(results)
    groups = by_query_group(results)
    out = settings.private_reports_dir
    run = {
        "timestamp": datetime.now(UTC).isoformat(),
        "split": split,
        "retrieval_only": retrieval_only,
        "commit": commit,
        "dirty": dirty,
        "cases": str(cases_path),
        "limit": limit,
        "n_cases": len(results),
        "n_failed": sum(r.error is not None for r in results),
        "summary": summary,
        "query_groups": groups,
        "named_labels": label_counts(results),
        "config": {
            "model": checker.model,
            "prompt": checker.prompt.version,
            "index_version": checker.index.version(),
            "index_replay": settings.index_replay_version is not None,
            "embed_model": settings.novelty_embed_model or "index default (bge-base)",
            **asdict(checker.params),
            "batch_size": checker.batch_size,
            "query_prompt": checker.query_prompt.version if checker.query_prompt else None,
            "query_model": checker.query_model if checker.query_prompt else None,
            "listwise_prompt": checker.listwise_prompt.version if checker.listwise_prompt else None,
            "listwise_model": checker.listwise_model if checker.listwise_prompt else None,
            "listwise_cap": checker.listwise_cap if checker.listwise_prompt else None,
            "first_stage": settings.novelty_first_stage,
            "pool_size": checker.params.pool_size,
        },
        "llm_cost_usd_new": checker.client.totals.fresh.cost_usd,
    }
    name = f"{EVAL_NAME}_{split}" + ("_retrieval" if retrieval_only else "")
    write_private(
        out / f"{name}.json",
        json.dumps({**run, "results": [r.model_dump() for r in results]}, indent=2, default=str)
        + "\n",
    )
    write_private(
        out / f"{name}.md", render_eval(results, summary, f"{cases_path} ({split})", groups)
    )
    append_private(out / EVAL_RUNS_LOG, json.dumps(run, default=str))
    mode = "retrieval-only " if retrieval_only else ""
    typer.echo(
        f"novelty eval {mode}[{split}] @ {commit[:10]}{' (dirty)' if dirty else ''}: "
        f"{len(results)} cases; recall@5 {summary['recall_at_5']:.3f}, "
        f"recall@10 {summary['recall_at_10']:.3f}, recall@20 {summary['recall_at_20']:.3f}, "
        f"pool recall {summary['pool_recall']:.3f}, "
        f"union recall {summary['union_recall']:.3f}"
    )
    return summary


class LLMCacheMissError(LLMAPIError):
    """A retrieval-only run needed an LLM response that is not cached."""


def _no_new_llm_calls(req: TransportRequest) -> TransportResponse:
    raise LLMCacheMissError(f"retrieval-only run: no cached {req.schema.__name__} response")


def _eval_case(
    case: NoveltyEvalCase, checker: Checker, settings: Settings, retrieval_only: bool = False
) -> NoveltyEvalCaseResult:
    try:
        checker.bind_eval_case(case.case_id)
        paper_qs, q_usage = checker.paper_queries(case.title, case.abstract)
        if retrieval_only:
            queries, pool, _, unpooled = checker.retrieve_query(
                case_query(case), case_target(case), paper_qs
            )
            return score_retrieval(case, queries, pool, unpooled)
        result = checker.check_query(
            case_query(case), case_target(case), 0, extra_queries=paper_qs, query_usage=q_usage
        )
    except (MissingAPIKeyError, LLMCacheMissError):
        raise
    except (IndexClientError, LLMError, RerankError, QueryGenerationError) as e:
        error = f"{type(e).__name__}: {e}"
        record_failure(settings.failures_file, "novelty", case.case_id, f"eval: {error}")
        return NoveltyEvalCaseResult(
            case_id=case.case_id,
            venue=case.venue,
            n_named=len(case.prior_work),
            n_unresolved=case.n_unresolved,
            n_after_cutoff=case.n_after_cutoff,
            found_at=[None] * len(case.prior_work),
            in_pool=[False] * len(case.prior_work),
            recall_at_5=0.0,
            recall_at_20=0.0,
            pool_recall=0.0,
            in_union=[False] * len(case.prior_work),
            found_by=[[] for _ in case.prior_work],
            error=error,
        )
    return score_case(case, result)


# --------------------------------------------------------------------------- first stage


def _recall(flags: list[bool]) -> float:
    return sum(flags) / len(flags) if flags else 0.0


def run_first_stage_eval(
    limit: int | None = None,
    cases_path: Path | None = None,
    settings: Settings | None = None,
    index: IndexAPI | None = None,
    client: LLMClient | None = None,
    pool_sizes: Sequence[int] = (),
) -> dict[str, float]:
    """Retrieval only, dev only: pool recall of the current round-robin pool vs the first-stage
    ranker (hand-set rule, and logistic regression with 5-fold CV grouped by submission) on
    search + /expand candidates. Fits the logistic model on all cases and saves it."""
    settings = settings or get_settings()
    cases_path = eval_cases_path(settings, "dev", cases_path)
    if not cases_path.exists():
        logger.error("no eval cases at %s", cases_path)
        raise typer.Exit(code=1)
    commit, dirty = git_state()
    cases = sample_cases(load_cases(cases_path), limit)
    checker = _checker(settings, index, client or LLMClient(settings, transport=_no_new_llm_calls))
    size = settings.novelty_pool_size

    tables: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    stage_cands: list[list[StageCandidate]] = []
    kept: list[NoveltyEvalCase] = []
    current_pool: list[bool] = []
    current_union: list[bool] = []
    stage_union: list[bool] = []
    n_cands: list[int] = []
    n_expand_new: list[int] = []
    failed: list[str] = []
    try:
        for case in cases:
            try:
                paper_qs, _ = checker.paper_queries(case.title, case.abstract)
                queries = baseline_queries(case_query(case)) + paper_qs
                target = case_target(case)
                pool, _, unpooled = retrieve_multi(
                    queries, checker.index, target, checker.params, case_query(case).statement
                )
                cands, counts = gather_candidates(queries, checker.index, target, checker.params)
            except (IndexClientError, LLMError, QueryGenerationError) as e:
                if isinstance(e, LLMCacheMissError):
                    raise
                record_failure(
                    settings.failures_file, "novelty", case.case_id, f"first-stage eval: {e}"
                )
                failed.append(case.case_id)
                continue
            papers = [c.candidate.paper for c in cands]
            for ref in case.prior_work:
                current_pool.append(any(matches(ref, c.paper) for c in pool))
                current_union.append(any(matches(ref, c.paper) for c in pool + unpooled))
                stage_union.append(any(matches(ref, p) for p in papers))
            y = np.array([any(matches(ref, p) for ref in case.prior_work) for p in papers], float)
            tables.append(feature_matrix(cands))
            labels.append(y)
            stage_cands.append(cands)
            kept.append(case)
            n_cands.append(len(cands))
            n_expand_new.append(counts.expansion_new)
    except LLMCacheMissError as e:
        logger.error("%s", e)
        raise typer.Exit(code=1) from e

    def pool_flags(scores: list[np.ndarray], n: int = size) -> list[bool]:
        flags = []
        for case, cands, sc in zip(kept, stage_cands, scores, strict=True):
            chosen = [cands[i].candidate.paper for i in select_pool(cands, sc, n)]
            flags += [any(matches(ref, p) for p in chosen) for ref in case.prior_work]
        return flags

    rule_scores = [rule_score(t) if len(t) else np.zeros(0) for t in tables]
    fold_models = fit_folds(tables, labels)
    cv_scores = cross_validated_scores(tables, labels, models=fold_models)
    rule_flags = pool_flags(rule_scores)
    learned_flags = pool_flags(cv_scores)
    by_size = {
        n: {
            "rule_pool_recall": _recall(pool_flags(rule_scores, n)),
            "learned_pool_recall_cv": _recall(pool_flags(cv_scores, n)),
            "rerank_calls": math.ceil(n / settings.novelty_batch_size),
        }
        for n in sorted({size, *pool_sizes})
    }
    full = fit_logistic(np.vstack([t for t in tables if len(t)]), np.concatenate(labels))
    settings.first_stage_model_file.parent.mkdir(parents=True, exist_ok=True)
    settings.first_stage_model_file.write_text(full.to_json() + "\n", encoding="utf-8")
    cv = CrossValidatedScorer(
        models=fold_models, fold_of={c.case_id: i % len(fold_models) for i, c in enumerate(kept)}
    )
    settings.first_stage_cv_file.write_text(cv.to_json() + "\n", encoding="utf-8")

    summary = {
        "current_pool_recall": _recall(current_pool),
        "current_union_recall": _recall(current_union),
        "stage_union_recall": _recall(stage_union),
        "rule_pool_recall": _recall(rule_flags),
        "learned_pool_recall_cv": _recall(learned_flags),
    }
    run = {
        "type": "first_stage_eval",
        "timestamp": datetime.now(UTC).isoformat(),
        "split": "dev",
        "commit": commit,
        "dirty": dirty,
        "cases": str(cases_path),
        "limit": limit,
        "n_cases": len(kept),
        "n_failed": len(failed),
        "n_named": len(current_pool),
        "summary": summary,
        "by_pool_size": by_size,
        "median_candidates": float(np.median(n_cands)) if n_cands else 0.0,
        "median_expand_new": float(np.median(n_expand_new)) if n_expand_new else 0.0,
        "pool_size": size,
        "config": {
            "index_version": checker.index.version(),
            "embed_model": settings.novelty_embed_model or "index default (bge-base)",
            "expand_seeds": checker.params.expand_seeds,
            "expand_max_candidates": checker.params.expand_max_candidates,
        },
        "learned_model_all_cases": json.loads(full.to_json()),
    }
    out = settings.private_reports_dir
    write_private(out / "novelty_first_stage_dev.json", json.dumps(run, indent=2) + "\n")
    write_private(out / "novelty_first_stage_dev.md", render_first_stage(run))
    append_private(out / EVAL_RUNS_LOG, json.dumps(run))
    typer.echo(
        f"first stage [dev] @ {commit[:10]}{' (dirty)' if dirty else ''}: {len(kept)} cases, "
        f"{len(current_pool)} named; pool recall current {summary['current_pool_recall']:.3f}, "
        f"rule {summary['rule_pool_recall']:.3f}, learned (5-fold CV) "
        f"{summary['learned_pool_recall_cv']:.3f}; union current "
        f"{summary['current_union_recall']:.3f}, search+expand {summary['stage_union_recall']:.3f}"
    )
    return summary


# --------------------------------------------------------------------------- negatives


def run_negatives(
    limit: int | None = 50,
    cases_path: Path | None = None,
    settings: Settings | None = None,
    index: IndexAPI | None = None,
    client: LLMClient | None = None,
) -> dict[str, int]:
    """Dev only: run the default pipeline on accepted submissions reviewers found novel and
    report the verdict distribution, with every `already_done` verdict spelled out."""
    settings = settings or get_settings()
    cases_path = cases_path or settings.novelty_eval_dir / "negatives_dev.jsonl"
    if "test" in cases_path.name:
        raise typer.BadParameter(f"{cases_path} looks like a test split; negatives are dev only")
    if not cases_path.exists():
        logger.error("no negatives at %s", cases_path)
        raise typer.Exit(code=1)
    rows = [
        NegativeCaseRow.model_validate_json(line)
        for line in cases_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cases = sample_cases(
        [
            NoveltyEvalCase(
                case_id=r.submission_id,
                venue=r.venue,
                title=r.submission_title,
                abstract=r.submission_abstract,
                cutoff_date=r.cutoff_date,
                prior_work=[],
            )
            for r in rows
        ],
        limit,
    )
    commit, dirty = git_state()
    checker = _checker(settings, index, client)
    results: list[tuple[NoveltyEvalCase, ClaimNovelty]] = []
    failed: list[str] = []
    try:
        for case in cases:
            try:
                checker.bind_eval_case(case.case_id)
                paper_qs, q_usage = checker.paper_queries(case.title, case.abstract)
                result = checker.check_query(
                    case_query(case),
                    case_target(case),
                    0,
                    extra_queries=paper_qs,
                    query_usage=q_usage,
                )
            except (IndexClientError, LLMError, RerankError, QueryGenerationError) as e:
                if isinstance(e, MissingAPIKeyError):
                    raise
                record_failure(settings.failures_file, "novelty", case.case_id, f"negatives: {e}")
                failed.append(case.case_id)
                continue
            results.append((case, result))
    except MissingAPIKeyError as e:
        logger.error("%s", e)
        raise typer.Exit(code=1) from e
    counts = {v.value: 0 for v in NoveltyVerdict}
    for _, r in results:
        counts[r.verdict.value] += 1
    run = {
        "type": "negatives",
        "timestamp": datetime.now(UTC).isoformat(),
        "split": "dev",
        "commit": commit,
        "dirty": dirty,
        "cases": str(cases_path),
        "limit": limit,
        "n_cases": len(results),
        "n_failed": len(failed),
        "verdicts": counts,
        "config": {
            "first_stage": settings.novelty_first_stage,
            "pool_size": checker.params.pool_size,
            "rerank_prompt": checker.prompt.version,
            "listwise_cap": checker.listwise_cap if checker.listwise_prompt else None,
            "index_version": checker.index.version(),
        },
        "llm_cost_usd_new": checker.client.totals.fresh.cost_usd,
    }
    out = settings.private_reports_dir
    write_private(
        out / "novelty_negatives_dev.json",
        json.dumps(
            {
                **run,
                "results": [
                    {"case": c.model_dump(mode="json"), "result": r.model_dump(mode="json")}
                    for c, r in results
                ],
            },
            indent=2,
        )
        + "\n",
    )
    write_private(out / "novelty_negatives_dev.md", render_negatives(run, results))
    append_private(out / EVAL_RUNS_LOG, json.dumps(run))
    typer.echo(
        f"negatives [dev] @ {commit[:10]}: {len(results)} submissions ({len(failed)} failed); "
        + ", ".join(f"{k} {v}" for k, v in counts.items())
    )
    return counts


# --------------------------------------------------------------------------- severity eval


def run_severity_eval(
    limit: int = 50,
    settings: Settings | None = None,
    index: IndexAPI | None = None,
    client: LLMClient | None = None,
    positives_path: Path | None = None,
    negatives_path: Path | None = None,
) -> dict:
    """Dev only: paper-level severity on the positive eval cases vs the negatives; AUC,
    distributions, and every negative scoring 3."""
    settings = settings or get_settings()
    run_eval(
        "dev", limit=limit, cases_path=positives_path, settings=settings, index=index, client=client
    )
    pos_run = json.loads((settings.private_reports_dir / "novelty_eval_dev.json").read_text())
    run_negatives(limit, cases_path=negatives_path, settings=settings, index=index, client=client)
    neg_run = json.loads((settings.private_reports_dir / "novelty_negatives_dev.json").read_text())
    pos = [r["severity"]["score"] for r in pos_run["results"] if r.get("severity")]
    neg = [
        x["result"]["severity"]["score"] for x in neg_run["results"] if x["result"].get("severity")
    ]
    dist = {
        "positives": {s: pos.count(s) for s in range(4)},
        "negatives": {s: neg.count(s) for s in range(4)},
    }
    neg3 = []
    for x in neg_run["results"]:
        sev = x["result"].get("severity")
        if not sev or sev["score"] < 3:
            continue
        titles = {
            j["candidate"]["paper"]["key"]: j["candidate"]["paper"]["title"]
            for j in x["result"]["ranked"][:10]
        }
        neg3.append(
            {
                "case_id": x["case"]["case_id"],
                "title": x["case"]["title"],
                "drivers": [f"{titles.get(d, '?')} ({d})" for d in sev["drivers"]],
                "reason": sev["reason"],
            }
        )
    summary = {
        "type": "severity_eval",
        "timestamp": datetime.now(UTC).isoformat(),
        "split": "dev",
        "commit": pos_run["commit"],
        "dirty": pos_run["dirty"],
        "index_replay": settings.index_replay_version,
        "n_positives": len(pos),
        "n_negatives": len(neg),
        "auc": auc(pos, neg),
        "distributions": dist,
        "negatives_scoring_3": neg3,
        "positive_recall": pos_run["summary"],
    }
    out = settings.private_reports_dir
    write_private(out / "novelty_severity_dev.json", json.dumps(summary, indent=2) + "\n")
    write_private(out / "novelty_severity_dev.md", render_severity(summary))
    append_private(out / EVAL_RUNS_LOG, json.dumps(summary))
    typer.echo(
        f"severity [dev]: AUC {summary['auc']:.3f} over {len(pos)} positives / {len(neg)} "
        f"negatives; negatives scoring 3: {len(neg3)}"
    )
    return summary
