"""find_prior_work: search + /expand + the shipped first-stage ranker (MCP_PLAN §3.1). No LLM."""

import logging
import re
import time
from datetime import date

from reviewer.config import Settings
from reviewer.mcp.bibtex import bibtex, unique_keys
from reviewer.mcp.index import HostedIndex
from reviewer.models import FindPriorWorkResult, IndexPaperDetail, PriorWorkCandidate, SearchQuery
from reviewer.novelty.firststage import (
    LogisticModel,
    feature_matrix,
    gather_candidates,
    select_pool,
)
from reviewer.novelty.retrieve import RetrievalParams, Target

logger = logging.getLogger(__name__)

MIN_QUERIES, MAX_QUERIES = 1, 12
MIN_WORDS, MAX_WORDS = 3, 60
MAX_K = 40
ABSTRACT_CHARS = 1200
PARAMS = RetrievalParams(query_k=30, expand_seeds=20, expand_max_candidates=500)


class QueryError(ValueError):
    """Invalid tool input; the message tells the model how to fix it."""


def validate(queries: list[str], k: int) -> list[str]:
    qs = [" ".join(q.split()) for q in queries if q and q.strip()]
    if not MIN_QUERIES <= len(qs) <= MAX_QUERIES:
        raise QueryError(
            f"Pass 6 to 10 component queries (got {len(qs)}): one per technique, objective, "
            "data-construction step or problem framing, phrased as a prior paper's abstract "
            "would describe it."
        )
    for i, q in enumerate(qs, 1):
        n = len(q.split())
        if not MIN_WORDS <= n <= MAX_WORDS:
            raise QueryError(
                f"Query {i} has {n} words; write each as a declarative sentence of 10 to 40 "
                "words, the way an abstract would describe that component."
            )
    if not 1 <= k <= MAX_K:
        raise QueryError(f"k must be between 1 and {MAX_K}.")
    return qs


def _query_index(query_id: str) -> int | None:
    m = re.fullmatch(r"q(\d+)", query_id)
    return int(m.group(1)) if m else None


def _link(d: IndexPaperDetail) -> str | None:
    if d.arxiv_id:
        return f"https://arxiv.org/abs/{d.arxiv_id}"
    if d.doi:
        return f"https://doi.org/{d.doi}"
    return None


def run_find_prior_work(
    index: HostedIndex,
    queries: list[str],
    before: date | None = None,
    k: int = MAX_K,
    settings: Settings | None = None,
) -> FindPriorWorkResult:
    qs = validate(queries, k)
    model = LogisticModel.load((settings or Settings(_env_file=None)).first_stage_model_file)
    search_queries = [
        SearchQuery(id=f"q{i}", origin="claim", kind="component", text=q)
        for i, q in enumerate(qs, 1)
    ]
    t0 = time.monotonic()
    index.prefetch_searches(qs, PARAMS.query_k, before)
    t1 = time.monotonic()
    cands, _counts = gather_candidates(search_queries, index, Target(cutoff=before), PARAMS)
    t2 = time.monotonic()
    scores = model(feature_matrix(cands)) if cands else []
    chosen = select_pool(cands, scores, k) if cands else []
    picked = [cands[i] for i in chosen]
    meta = index.papers([c.candidate.paper.key for c in picked])
    t3 = time.monotonic()
    logger.info(
        "find_prior_work: %d queries; searches %.1fs, expand+features %.1fs, metadata %.1fs",
        len(qs),
        t1 - t0,
        t2 - t1,
        t3 - t2,
    )
    details = [
        meta.get(c.candidate.paper.key)
        or IndexPaperDetail(
            key=c.candidate.paper.key,
            kind=c.candidate.paper.kind,
            title=c.candidate.paper.title,
            year=c.candidate.paper.year,
            abstract=c.candidate.paper.abstract,
        )
        for c in picked
    ]
    keys = unique_keys(details)
    out = []
    for rank, (i, sc, d, key) in enumerate(zip(chosen, picked, details, keys, strict=True), 1):
        paper = sc.candidate.paper
        abstract = d.abstract or paper.abstract
        matched = sorted(
            {n for h in sc.candidate.hits if (n := _query_index(h.query_id)) is not None}
        )
        out.append(
            PriorWorkCandidate(
                rank=rank,
                key=paper.key,
                title=d.title or paper.title,
                year=d.year or paper.year,
                venue=d.venue,
                authors=d.authors,
                abstract=abstract[:ABSTRACT_CHARS] if abstract else None,
                link=_link(d),
                seed_count=sc.seed_count,
                matched_queries=matched,
                score=round(float(scores[i]), 4),
                bibtex=bibtex(d, key),
            )
        )
    return FindPriorWorkResult(
        queries=qs,
        before=before,
        index_version=index.version(),
        candidates=out,
        n_considered=len(cands),
    )
