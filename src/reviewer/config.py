"""Settings loaded from the environment and `.env`."""

from functools import lru_cache
from pathlib import Path
from typing import Any, Self

from pydantic import Field, PrivateAttr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGED_NOVELTY_MODELS = Path(__file__).resolve().parent / "novelty" / "models"
"""The committed first-stage ranker (shipped in the wheel); `reviewer novelty first-stage-eval`
rewrites it here."""

# USD per million tokens: (input, output). Cache reads/writes are billed at input rates here
# as an upper bound; the per-model rows come from Anthropic's published price list.
MODEL_PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


class Settings(BaseSettings):
    """Runtime configuration. Every path is absolute and derived from `data_dir` by default."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    anthropic_api_key: str | None = None
    grobid_url: str = "http://localhost:8070"
    grobid_timeout_s: float = 300.0
    llm_model: str = "claude-sonnet-5"
    llm_model_strong: str = "claude-opus-5-5"
    llm_effort: str = "medium"
    llm_max_tokens: int = 32000
    llm_max_retries: int = 2
    llm_timeout_s: float = 180.0
    """Read timeout for one streamed LLM call: no bytes for this long aborts it (the SDK's
    own retries then apply). Guards against a connection that hangs indefinitely."""
    extract_prompt: str = "extract_v3"
    critic_prompt: str = "critic_v4"
    number_grounding: bool = True
    """After the critic, check every claim number against the paper's full text; send
    ungrounded numbers back to the critic."""
    grounding_max_rounds: int = 2
    verify_prompt: str | None = None
    """Per-claim verification pass after the critic (e.g. 'critic_v5'); None: off."""
    table_char_budget: int = 1500
    """Max characters of cell/raw text sent per table (~400 tokens); longer tables are cut."""
    duplicate_threshold: float = 0.6
    arxiv_delay_s: float = 3.0
    user_agent: str = "reviewer/0.1 (https://github.com/Rishab-Ghosh/reviewer; research tool)"
    # Citation checker
    openalex_api_key: str | None = None
    openalex_email: str | None = None
    """Sent as `mailto` to OpenAlex and Crossref (polite pools)."""
    openalex_min_remaining_usd: float = 0.02
    """Stop paid OpenAlex title searches when the key's remaining daily budget reaches this;
    the key is shared with the reviewer-index daily job."""
    citation_prompt: str = "support_v5"
    citation_llm_workers: int = 6
    citation_ref_workers: int = 4
    citation_abstract_chars: int = 3000
    """Abstracts longer than this are cut before being shown to the support judge."""

    judge_prompt: str = "judge_v2"
    judge_model: str = "claude-opus-5-5"
    writing_prompt: str = "writing_v1"
    writing_model: str = "claude-opus-5-5"
    writing_budget_usd: float = 10.0
    """`reviewer writing` stops before a new call once this run's fresh spend reaches this."""
    methods_model: str = "claude-opus-5-5"
    methods_prompt: str = "methods_v2"
    methods_effort: str = "medium"
    """Effort for the methodology checker call (the repair, matcher and verifier calls too)."""
    meta_model: str = "claude-opus-5-5"
    meta_prompt: str = "meta_v1"
    meta_k: int = 1
    """Independent judgment runs per paper; 1 for the lite configuration, 3 for full (§4.1)."""
    meta_novelty_top_claims: int | None = 2
    """Lite config: run novelty on only the paper's top N claims (by confidence); None = all."""
    meta_budget_usd: float = 120.0
    """Hard cap on this run's total fresh spend across all checkers + meta judgment (§10)."""
    meta_run_baselines: bool = True
    """Also run the checker-free baselines (a)/(b) and the contamination probe (§4, §6, §9 step
    5) alongside the meta-reviewer's own judgment call. Cheap; counted against meta_budget_usd."""
    meta_blind_decision_status: bool = False
    """Dev runs on OpenReview papers: drop format findings that reveal camera-ready status
    (anonymity, page limits) from the meta-reviewer's evidence (see evidence.py)."""
    baseline_abstract_prompt: str = "baseline_abstract_v1"
    baseline_full_prompt: str = "baseline_full_v1"
    contamination_prompt: str = "contamination_v1"
    methods_repair: bool = True
    methods_runs: int = 2
    """`reviewer methods check` runs the checker this many times per paper: 2 (default) reports
    findings both runs raise as confirmed and one-run findings as possible; 1 = a single run."""
    methods_eval_dir: Path = Field(
        default=PROJECT_ROOT.parent / "reviewer-index" / "data" / "methodology_eval"
    )
    """reviewer-index methodology-concern dataset: dev.jsonl (all tuning); test.jsonl is locked."""
    openreview_username: str | None = None
    openreview_password: str | None = None
    """Only needed to download submission PDFs the first time (then cached)."""
    openreview_delay_s: float = 7.5
    """Spacing between OpenReview requests. The observed limit on 2026-09-26 was stricter
    (18 requests, then a ~16-minute lockout); 429s are waited out (openreview_pdf)."""
    openreview_v2_history: bool = False
    """Query API v2 edit histories. Off: they returned no edits to our account on every paper
    tried (2026-09-26), so each query only costs a rate-limited request."""
    """Send findings whose quotes all fail verification back once for corrected quotes."""
    index_url: str = "http://127.0.0.1:8765"
    index_api_key: str | None = Field(default=None, repr=False)
    """X-API-Key for a public-mode index (the MCP server sets it); sent by the citation
    checker's index client. Never logged."""
    """reviewer-index query API (`reviewer-index serve`)."""
    index_timeout_s: float = 60.0
    index_replay_version: str | None = None
    """Replay mode: use this index version for cache keys and serve index responses only from
    the cache (a miss is an error), so an earlier run's retrieval is reproduced exactly."""
    novelty_first_stage: str = "learned"
    """How the rerank pool is filled: 'roundrobin' (per-query rank, /neighbors expansion),
    'rule' or 'learned' (search + POST /expand candidates scored by the first-stage ranker), or
    'learned_cv' (dev eval only: each case scored by the fold model that did not see it)."""
    novelty_expand_seeds: int = 20
    novelty_expand_max_candidates: int = 2000
    novelty_embed_model: str | None = None
    """Index embedding space for /search (e.g. 'bge-large'); unset = the index default
    (bge-base). /neighbors similarities always use the index default."""
    novelty_prompt: str = "rerank_v3"
    novelty_severity: bool = True
    """Score each claim's overlap 0-3 with one LLM call (severity_v1) and derive the verdict from
    it (0 novel, 1-2 incremental, 3 already done); False keeps the label rule."""
    novelty_severity_prompt: str = "severity_v2"
    novelty_severity_model: str = "claude-opus-5-5"
    novelty_search_k: int = 50
    """Hits per search query (statement; method + problem)."""
    novelty_expand_top: int = 10
    """Top search candidates (by similarity) expanded one hop through /neighbors."""
    novelty_neighbors_per_node: int = 20
    """Most similar neighbors kept per expanded candidate."""
    novelty_pool_size: int = 150
    """Candidates sent to the LLM for judgment per claim, filled round-robin by query rank."""
    novelty_multi_query: bool = True
    """Add LLM-generated component queries (claim-level and paper-level) to the baseline two."""
    novelty_query_prompt: str = "queries_v1"
    novelty_query_model: str = "claude-opus-5-5"
    novelty_listwise: bool = True
    """After batch labelling, order the top same/subsumes/close candidates with one listwise
    call; the top 5 comes from that order instead of label + embedding similarity."""
    novelty_listwise_cap: int = 60
    novelty_listwise_prompt: str = "listwise_v1"
    novelty_listwise_model: str = "claude-opus-5-5"
    novelty_query_k: int = 30
    """Hits per generated query."""
    novelty_batch_size: int = 20
    """Candidates per rerank call."""
    novelty_rerank_workers: int = 8
    """Rerank batches of one claim judged concurrently (results combined in batch order)."""
    novelty_claim_workers: int = 4
    """Claims of one paper checked concurrently (results kept in claim order)."""
    novelty_eval_dir: Path = Field(
        default=PROJECT_ROOT.parent / "reviewer-index" / "data" / "novelty_eval"
    )
    """Holds dev.jsonl (all tuning) and test.jsonl (only on explicit request, logged)."""
    meta_decisions_file: Path = Field(
        default=PROJECT_ROOT.parent / "reviewer-index" / "data" / "openreview" / "decisions.jsonl"
    )
    """ICLR 2024/25 decisions and every reviewer's rating: the meta dev run's ground truth."""

    cache_root: Path | None = None
    """Shared cache for public lookups (index, arXiv, OpenAlex, Crossref); default
    `data_dir/cache`. The web worker sets `/srv/data/cache/public`."""
    paper_cache_dir: Path | None = None
    """Cache for paper-content calls (LLM, GROBID TEI, Docling); default `cache_dir`. The web
    worker roots it per paper at `/srv/data/cache/papers/<paper_id>` (BP §4, §7)."""
    novelty_model_dir: Path | None = None
    """Where the first-stage ranker files live; default `PACKAGED_NOVELTY_MODELS` (inside the
    package, so the MCP wheel ships them). Tests point it at a temp dir."""
    parse_docling_tables: bool = True
    """Pick Docling tables when better than GROBID's. The web profile turns it off (BP cut 4)."""
    llm_retry_truncated: bool = False
    """On a `max_tokens` stop, retry the call once at 2x `llm_max_tokens` (BP §4; web profile)."""
    healthchecks_index_update_url: str | None = None
    """healthchecks.io ping URL of the nightly index update; `deadline-guard` pings it on a
    skip so a skipped night is not reported as missing (D58)."""

    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    reports_dir: Path = Field(default=PROJECT_ROOT / "reports")

    @property
    def arxiv_dir(self) -> Path:
        return self.data_dir / "papers" / "arxiv"

    @property
    def own_dir(self) -> Path:
        return self.data_dir / "papers" / "own"

    @property
    def cache_dir(self) -> Path:
        return self.cache_root or self.data_dir / "cache"

    @property
    def paper_cache(self) -> Path:
        """Root for caches that hold paper content (see `paper_cache_dir`)."""
        return self.paper_cache_dir or self.cache_dir

    _call_guard: Any = PrivateAttr(default=None)
    """Runtime only (never from env): `guard(TransportRequest) -> ContextManager[Slot | None]`
    wrapped around every uncached LLM call; the web orchestrator's budget and step deadline."""

    @property
    def call_guard(self) -> Any:
        return self._call_guard

    def with_call_guard(self, guard: Any) -> Self:
        """A copy whose LLM clients run every uncached call inside `guard` (copies keep it)."""
        out = self.model_copy()
        out._call_guard = guard
        return out

    @property
    def parsed_dir(self) -> Path:
        return self.data_dir / "parsed"

    @property
    def claims_dir(self) -> Path:
        return self.data_dir / "claims"

    @property
    def gold_dir(self) -> Path:
        return self.data_dir / "gold"

    @property
    def private_dir(self) -> Path:
        """Gitignored. Nothing derived from here may be committed."""
        return self.data_dir / "private"

    @property
    def private_gold_dir(self) -> Path:
        return self.private_dir / "gold"

    @property
    def critiques_dir(self) -> Path:
        return self.private_dir / "critiques"

    @property
    def private_reports_dir(self) -> Path:
        """Gitignored reports that include own-paper or private-gold content."""
        return self.reports_dir / "private"

    @property
    def novelty_dir(self) -> Path:
        """Per-paper novelty JSON. Private: lab-tier and own-paper results."""
        return self.private_reports_dir / "novelty"

    @property
    def meta_dir(self) -> Path:
        """Per-paper meta-reviews, one folder per meta prompt (private: no ratings/decisions ever
        go public from dev work)."""
        return self.private_reports_dir / "meta" / self.meta_prompt

    @property
    def first_stage_model_file(self) -> Path:
        """Logistic first-stage ranker fitted on the dev split (`novelty first-stage-eval`)."""
        return (self.novelty_model_dir or PACKAGED_NOVELTY_MODELS) / "first_stage_lr.json"

    @property
    def first_stage_cv_file(self) -> Path:
        """Fold models + fold of each dev case (NOVELTY_FIRST_STAGE=learned_cv, dev eval only)."""
        return (self.novelty_model_dir or PACKAGED_NOVELTY_MODELS) / "first_stage_lr_cv.json"

    @property
    def judge_dir(self) -> Path:
        """Per-paper claim judgments for the current judge prompt (private: include own and lab
        papers); one subdirectory per prompt version."""
        return self.private_reports_dir / "judge" / self.judge_prompt

    @property
    def methods_dir(self) -> Path:
        """Per-paper methodology reviews (private: lab and own papers)."""
        return self.private_reports_dir / "methods" / self.methods_prompt

    @property
    def openreview_pdf_dir(self) -> Path:
        """Submission PDFs as reviewers saw them (gitignored)."""
        return self.data_dir / "openreview" / "pdf"

    @property
    def openreview_parsed_dir(self) -> Path:
        """ParsedPaper JSON for OpenReview submissions, `or-<submission id>.json` (gitignored)."""
        return self.data_dir / "openreview" / "parsed"

    @property
    def methods_review_eval_dir(self) -> Path:
        """Checker reviews of dev submissions (private), one subdirectory per prompt."""
        return self.private_reports_dir / "methods_review_eval" / self.methods_prompt

    @property
    def calibration_file(self) -> Path:
        return self.private_reports_dir / "calibration_10.md"

    @property
    def corpus_file(self) -> Path:
        return self.data_dir / "corpus.yaml"

    @property
    def failures_file(self) -> Path:
        return self.data_dir / "failures.jsonl"


@lru_cache
def get_settings() -> Settings:
    return Settings()
