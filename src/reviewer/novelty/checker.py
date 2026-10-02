"""Check one claim (or every claim of a paper) against the index."""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from reviewer.claims.prompts import Prompt
from reviewer.config import Settings
from reviewer.llm.client import LLMClient
from reviewer.models import (
    Candidate,
    CandidateCounts,
    Claim,
    ClaimNovelty,
    NoveltyQuery,
    PaperClaims,
    PaperNovelty,
    SearchQuery,
    TokenUsage,
)
from reviewer.novelty.firststage import (
    CrossValidatedScorer,
    LogisticModel,
    Scorer,
    retrieve_first_stage,
    rule_score,
)
from reviewer.novelty.index_client import IndexAPI
from reviewer.novelty.queries import claim_queries, paper_queries
from reviewer.novelty.rerank import (
    TOP_N,
    judge_all,
    listwise_rerank,
    load_rerank_prompt,
    rank,
    verdict,
)
from reviewer.novelty.retrieve import RetrievalParams, Target, baseline_queries, retrieve_multi
from reviewer.novelty.severity import score_severity, severity_verdict

logger = logging.getLogger(__name__)


@dataclass
class Checker:
    index: IndexAPI
    client: LLMClient
    prompt: Prompt
    model: str
    params: RetrievalParams
    batch_size: int
    max_retries: int
    rerank_workers: int = 1
    claim_workers: int = 1
    query_prompt: Prompt | None = None
    """None: baseline queries only (v0). Otherwise generated component queries are added."""
    query_model: str = ""
    listwise_prompt: Prompt | None = None
    """None: final order is label, then embedding similarity (no listwise call)."""
    listwise_model: str = ""
    listwise_cap: int = 60
    severity_prompt: Prompt | None = None
    """None: rule verdict from the top-5 labels. Otherwise an LLM severity score (0-3)."""
    severity_model: str = ""
    first_stage_cv: CrossValidatedScorer | None = None
    """Fold models for the dev cases the learned ranker was fitted on: a dev eval scores those
    cases with the fold model trained without them (learned and learned_cv)."""
    first_stage_default: Scorer | None = None
    """The scorer for everything else (the model fitted on all CV cases, for `learned`)."""
    first_stage: Scorer | None = None
    """None: round-robin pool with /neighbors expansion. Otherwise search + /expand candidates
    ranked by this scorer."""

    def __post_init__(self) -> None:
        self.first_stage_default = self.first_stage

    def bind_eval_case(self, case_id: str) -> None:
        """Use the fold model for a dev case the learned ranker was fitted on; otherwise the
        default scorer (learned_cv refuses cases outside the CV set)."""
        if self.first_stage_cv is not None and case_id in self.first_stage_cv.fold_of:
            self.first_stage = self.first_stage_cv.for_case(case_id)
        elif self.first_stage_default is _unbound_cv_scorer:
            raise ValueError(f"case {case_id} was not in the first-stage CV set")
        else:
            self.first_stage = self.first_stage_default

    @classmethod
    def from_settings(
        cls, settings: Settings, index: IndexAPI, client: LLMClient, prompt: Prompt, model: str
    ) -> "Checker":
        return cls(
            index=index,
            client=client,
            prompt=prompt,
            model=model,
            first_stage=first_stage_scorer(settings),
            first_stage_cv=(
                CrossValidatedScorer.load(settings.first_stage_cv_file)
                if settings.novelty_first_stage in ("learned", "learned_cv")
                and settings.first_stage_cv_file.exists()
                else None
            ),
            params=RetrievalParams(
                expand_seeds=settings.novelty_expand_seeds,
                expand_max_candidates=settings.novelty_expand_max_candidates,
                search_k=settings.novelty_search_k,
                query_k=settings.novelty_query_k,
                expand_top=settings.novelty_expand_top,
                neighbors_per_node=settings.novelty_neighbors_per_node,
                pool_size=settings.novelty_pool_size,
            ),
            batch_size=settings.novelty_batch_size,
            max_retries=settings.llm_max_retries,
            rerank_workers=settings.novelty_rerank_workers,
            claim_workers=settings.novelty_claim_workers,
            query_prompt=(
                load_rerank_prompt(settings.novelty_query_prompt)
                if settings.novelty_multi_query
                else None
            ),
            query_model=settings.novelty_query_model,
            listwise_prompt=(
                load_rerank_prompt(settings.novelty_listwise_prompt)
                if settings.novelty_listwise
                else None
            ),
            listwise_model=settings.novelty_listwise_model,
            listwise_cap=settings.novelty_listwise_cap,
            severity_prompt=(
                load_rerank_prompt(settings.novelty_severity_prompt)
                if settings.novelty_severity
                else None
            ),
            severity_model=settings.novelty_severity_model,
        )

    def paper_queries(self, title: str, abstract: str) -> tuple[list[SearchQuery], TokenUsage]:
        if self.query_prompt is None:
            return [], TokenUsage()
        return paper_queries(
            self.client,
            title,
            abstract,
            prompt=self.query_prompt,
            model=self.query_model,
            max_retries=self.max_retries,
        )

    def claim_queries(self, query: NoveltyQuery) -> tuple[list[SearchQuery], TokenUsage]:
        if self.query_prompt is None:
            return [], TokenUsage()
        return claim_queries(
            self.client,
            query,
            prompt=self.query_prompt,
            model=self.query_model,
            max_retries=self.max_retries,
        )

    def retrieve_query(
        self, query: NoveltyQuery, target: Target, extra_queries: list[SearchQuery] | None = None
    ) -> tuple[list[SearchQuery], list[Candidate], CandidateCounts, list[Candidate]]:
        """Retrieval only: (queries, pool, counts, unpooled); no LLM judgment."""
        queries = baseline_queries(query) + (extra_queries or [])
        if self.first_stage is None:
            pool, counts, unpooled = retrieve_multi(
                queries, self.index, target, self.params, query.statement
            )
        else:
            pool, counts, unpooled = retrieve_first_stage(
                queries, self.index, target, self.params, self.first_stage
            )
        return queries, pool, counts, unpooled

    def check_query(
        self,
        query: NoveltyQuery,
        target: Target,
        claim_index: int,
        extra_queries: list[SearchQuery] | None = None,
        query_usage: TokenUsage | None = None,
    ) -> ClaimNovelty:
        """Baseline queries + `extra_queries` (generated), then one-hop expansion, pool, rerank."""
        queries, pool, counts, unpooled = self.retrieve_query(query, target, extra_queries)
        judged, usage = judge_all(
            self.client,
            query,
            pool,
            prompt=self.prompt,
            model=self.model,
            batch_size=self.batch_size,
            max_retries=self.max_retries,
            workers=self.rerank_workers,
        )
        ranked = rank(judged)
        if self.listwise_prompt is not None:
            ranked, lw_usage = listwise_rerank(
                self.client,
                query,
                ranked,
                prompt=self.listwise_prompt,
                model=self.listwise_model,
                cap=self.listwise_cap,
                max_retries=self.max_retries,
            )
            usage += lw_usage
        top = ranked[:TOP_N]
        severity = None
        claim_verdict = verdict(top)
        if self.severity_prompt is not None:
            severity, s_usage = score_severity(
                self.client,
                query,
                ranked,
                prompt=self.severity_prompt,
                model=self.severity_model,
                max_retries=self.max_retries,
            )
            usage += s_usage
            claim_verdict = severity_verdict(severity.score)
        return ClaimNovelty(
            claim_index=claim_index,
            query=query,
            verdict=claim_verdict,
            severity=severity,
            top=top,
            ranked=ranked,
            counts=counts,
            token_usage=usage,
            queries=queries,
            unpooled=unpooled,
            query_usage=query_usage or TokenUsage(),
        )

    def check_paper(
        self, claims: PaperClaims, tier: str, target: Target, abstract: str
    ) -> PaperNovelty:
        from concurrent.futures import ThreadPoolExecutor

        paper_qs, paper_usage = self.paper_queries(claims.title, abstract)

        def one(i: int) -> tuple[ClaimNovelty, TokenUsage]:
            logger.info("%s claim %d/%d", claims.paper_id, i + 1, len(claims.claims))
            query = query_from_claim(claims.claims[i])
            claim_qs, claim_usage = self.claim_queries(query)
            q_usage = claim_usage + paper_usage if i == 0 else claim_usage
            return self.check_query(query, target, i, claim_qs + paper_qs, q_usage), q_usage

        n = len(claims.claims)
        with ThreadPoolExecutor(max_workers=max(1, min(self.claim_workers, n or 1))) as ex:
            outputs = list(ex.map(one, range(n)))  # claim order, whatever finishes first
        results = []
        usage = TokenUsage()
        for result, q_usage in outputs:
            results.append(result)
            usage += result.token_usage + q_usage
        return PaperNovelty(
            paper_id=claims.paper_id,
            title=claims.title,
            tier=tier,
            cutoff=target.cutoff.isoformat() if target.cutoff else None,
            cutoff_source=target.cutoff_source,
            index_version=self.index.version(),
            model=self.model,
            prompt_version="+".join(
                p.version
                for p in (
                    self.prompt,
                    self.query_prompt,
                    self.listwise_prompt,
                    self.severity_prompt,
                )
                if p
            ),
            timestamp=datetime.now(UTC),
            claims=results,
            token_usage=usage,
        )


def _unbound_cv_scorer(x: np.ndarray) -> np.ndarray:
    raise ValueError("learned_cv scores only dev eval cases (the eval binds the fold model)")


def first_stage_scorer(settings: Settings) -> Scorer | None:
    mode = settings.novelty_first_stage
    if mode == "roundrobin":
        return None
    if mode == "rule":
        return rule_score
    if mode == "learned":
        return LogisticModel.load(settings.first_stage_model_file)
    if mode == "learned_cv":
        return _unbound_cv_scorer
    raise ValueError(f"novelty_first_stage must be roundrobin, rule or learned, not {mode!r}")


def query_from_claim(claim: Claim) -> NoveltyQuery:
    return NoveltyQuery(
        statement=claim.statement,
        method=claim.method,
        problem=claim.problem,
        setting=claim.setting,
        result=claim.result,
    )
