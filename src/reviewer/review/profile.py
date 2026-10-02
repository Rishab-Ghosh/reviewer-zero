"""The web worker's Settings profile (MASTER_PLAN D67, BP cut 4, BP §4)."""

from pathlib import Path

from reviewer.config import Settings

WEB_OVERRIDES: dict[str, object] = {
    # D67: the meta step is the judgment only; no research baselines, one sample, no
    # OpenReview decision blinding.
    "meta_run_baselines": False,
    "meta_k": 1,
    "meta_blind_decision_status": False,
    # BP cut 4: GROBID tables only.
    "parse_docling_tables": False,
    # BP §4: a max_tokens stop is retried once at 2x, then recorded as a failure.
    "llm_retry_truncated": True,
}


NOVELTY_MODES: dict[str, dict[str, object]] = {
    # The 60 best first-stage candidates go to the LLM reranker: 3 rerank calls per claim
    # instead of 8 (the reranker dominates review time). Measured in MCP_PLAN M11.
    "lite": {"novelty_pool_size": 60},
    # The pipeline's pool, used for the 43.2% dev recall@10 (MEASURED_CLAIMS §1).
    "full": {"novelty_pool_size": 150},
}


def web_settings(base: Settings) -> Settings:
    return base.model_copy(update=WEB_OVERRIDES)


def paper_settings(
    base: Settings, work_dir: Path, public_cache: Path, paper_cache: Path
) -> Settings:
    """Per-paper settings: every checker output under `work_dir`, public lookups shared,
    paper-content caches (LLM, GROBID, Docling) rooted per paper."""
    return web_settings(base).model_copy(
        update={
            "data_dir": work_dir,
            "reports_dir": work_dir / "reports",
            "cache_root": public_cache,
            "paper_cache_dir": paper_cache,
        }
    )


def init_work_dir(s: Settings) -> None:
    """Create what the checkers expect to exist under a fresh per-paper data dir: the output
    dirs they write into without creating, and an empty corpus (the paper is passed by id)."""
    for d in (s.parsed_dir, s.claims_dir):
        d.mkdir(parents=True, exist_ok=True)
    if not s.corpus_file.exists():
        s.corpus_file.write_text("papers: []\n", encoding="utf-8")
