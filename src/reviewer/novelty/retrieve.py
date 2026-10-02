"""Candidate retrieval for one claim: two searches, union, one-hop expansion, pool. Pure apart
from the injected index client."""

import calendar
import logging
import re
from dataclasses import dataclass, replace
from datetime import date

from reviewer.models import (
    Candidate,
    CandidateCounts,
    CandidateSource,
    IndexPaper,
    NoveltyQuery,
    QueryHit,
    SearchQuery,
)
from reviewer.novelty.index_client import IndexAPI

logger = logging.getLogger(__name__)

_NEW_ARXIV_ID = re.compile(r"^(\d{2})(\d{2})\.(\d{4,5})$")
MIN_TITLE_WORDS_FOR_MATCH = 4


@dataclass(frozen=True)
class RetrievalParams:
    search_k: int = 50
    """Hits per baseline query."""
    query_k: int = 30
    """Hits per generated query."""
    expand_seeds: int = 20
    """First-stage ranker: top search results (round-robin order) sent to /expand as seeds."""
    expand_max_candidates: int = 2000
    expand_top: int = 10
    neighbors_per_node: int = 20
    pool_size: int = 60


@dataclass(frozen=True)
class Target:
    """The paper whose claim is checked: excluded from its own results, and nothing newer than
    `cutoff` counts as prior work."""

    arxiv_id: str | None = None
    title: str | None = None
    abstract: str = ""
    """Used only to catch the target's own earlier preprint under a different title
    (`is_own_preprint`); not required for the other filters."""
    cutoff: date | None = None
    """None = no cutoff (novel as of today). Sent to the index as `before`."""
    cutoff_source: str | None = None


def normalize_title(title: str | None) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).split())


def arxiv_month_end(arxiv_id: str) -> date | None:
    """Last day of the submission month of a new-style arXiv id (2504.15895 -> 2025-04-30)."""
    m = _NEW_ARXIV_ID.match(arxiv_id)
    if not m:
        return None
    year, month = 2000 + int(m.group(1)), int(m.group(2))
    return date(year, month, calendar.monthrange(year, month)[1])


def arxiv_month_start(arxiv_id: str) -> date:
    m = _NEW_ARXIV_ID.match(arxiv_id)
    assert m is not None
    return date(2000 + int(m.group(1)), int(m.group(2)), 1)


def target_for(paper_id: str, title: str | None, cutoff: bool, abstract: str = "") -> Target:
    """Target for a corpus paper: arXiv papers are cut off at their submission month; own papers
    (and `cutoff=False`) have no cutoff."""
    if paper_id.startswith("own-") or not cutoff:
        return Target(
            arxiv_id=None if paper_id.startswith("own-") else paper_id,
            title=title,
            abstract=abstract,
        )
    return Target(
        arxiv_id=paper_id,
        title=title,
        abstract=abstract,
        cutoff=arxiv_month_end(paper_id),
        cutoff_source="end of arXiv submission month",
    )


def resolve_v1_date(index: IndexAPI, target: Target) -> Target:
    """Replace a month-end cutoff by the paper's arXiv v1 date as the index records it (found by
    searching its title and matching the arXiv id). Unchanged when the index lacks it."""
    if not target.arxiv_id or target.cutoff is None or not target.title:
        return target
    for p in index.search(target.title, 10):
        if p.key.removeprefix("arxiv:") == target.arxiv_id and p.published_date:
            return replace(target, cutoff=p.published_date, cutoff_source="arXiv v1 date (index)")
    logger.warning(
        "v1 date of %s not found in the index; cutoff stays %s", target.arxiv_id, target.cutoff
    )
    return replace(target, cutoff_source="end of arXiv submission month (v1 date not in index)")


def _arxiv_order(arxiv_id: str) -> tuple[int, int] | None:
    m = _NEW_ARXIV_ID.match(arxiv_id)
    return (int(m.group(1) + m.group(2)), int(m.group(3))) if m else None


def is_self(paper: IndexPaper, target: Target) -> bool:
    if target.arxiv_id and paper.key.removeprefix("arxiv:") == target.arxiv_id:
        return True
    t = normalize_title(target.title)
    return len(t.split()) >= MIN_TITLE_WORDS_FOR_MATCH and normalize_title(paper.title) == t


def is_future(paper: IndexPaper, target: Target) -> bool:
    """True when `paper` appeared after the target. arXiv ids are ordered by submission, so two
    arXiv papers compare exactly; otherwise the published date, or the year when there is no
    date, is compared with the cutoff. Undated papers are handled by `is_undated`."""
    if target.cutoff is None:
        return False
    theirs = _arxiv_order(paper.key.removeprefix("arxiv:"))
    if target.arxiv_id:
        mine = _arxiv_order(target.arxiv_id)
        if mine and theirs:
            return theirs > mine
    if theirs and arxiv_month_start(paper.key.removeprefix("arxiv:")) > target.cutoff:
        # An arXiv id cannot predate the first day of its month, whatever date the index holds
        # (some index dates are earlier than the id allows).
        return True
    if paper.published_date is not None:
        return paper.published_date > target.cutoff
    if paper.year is not None:
        return paper.year > target.cutoff.year
    return False


def is_undated(paper: IndexPaper, target: Target) -> bool:
    """No way to compare `paper` with the cutoff: dropped whenever there is a cutoff."""
    arxiv_pair = target.arxiv_id and _arxiv_order(paper.key.removeprefix("arxiv:"))
    return not arxiv_pair and paper.published_date is None and paper.year is None


OWN_PREPRINT_JACCARD = 0.85
"""Threshold on word-set overlap (title or abstract) for `is_own_preprint`. Calibrated on 5
same-work cases found in a dev eval (arXiv preprint vs. the OpenReview submission it became):
title/abstract Jaccard of 0.79/0.96, 0.92/0.94, and 0.25/0.94 for three genuine self-matches
(one retitled for camera-ready, so title alone misses it) against 0.25/0.40 and 0.65/0.35 for
two unrelated-but-topically-close papers. 0.85 on either field separates them cleanly."""


def _word_set(text: str) -> set[str]:
    return set(re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).split())


def text_jaccard(a: str, b: str) -> float:
    wa, wb = _word_set(a), _word_set(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def is_own_preprint(
    paper: IndexPaper, target: Target, threshold: float = OWN_PREPRINT_JACCARD
) -> bool:
    """True when `paper` is very likely the target's own earlier preprint under a different
    title or framing (a revision, a renamed camera-ready version, a workshop draft), not
    independent prior work that happens to be close: title or abstract word overlap at or above
    `threshold`, and not newer than the target (so a coincidental match to a later, unrelated
    paper is never caught by this). This must hold regardless of label or verdict downstream: a
    paper's own preprint is never prior work, so it is dropped at retrieval, before any claim
    ever reaches the LLM."""
    if is_future(paper, target):
        return False
    return (
        text_jaccard(target.title or "", paper.title or "") >= threshold
        or text_jaccard(target.abstract, paper.abstract or "") >= threshold
    )


def method_problem_query(query: NoveltyQuery) -> str:
    parts = [p.strip().rstrip(".") for p in (query.method, query.problem) if p.strip()]
    return ". ".join(parts)


def has_metadata(paper: IndexPaper) -> bool:
    """False when the index withholds the title or abstract (known-bad metadata): such papers
    cannot be judged, so they are skipped."""
    return bool(paper.title and paper.title.strip() and paper.abstract and paper.abstract.strip())


def _admit(papers: list[IndexPaper], target: Target, counts: CandidateCounts) -> list[IndexPaper]:
    """Drop self and unjudgeable papers. The index already applies the cutoff (`before`); the
    date checks here are a guard, so `filtered_future`/`filtered_undated` should stay 0."""
    out = []
    for p in papers:
        if not has_metadata(p):
            counts.filtered_no_metadata += 1
        elif is_self(p, target):
            counts.filtered_self += 1
        elif is_own_preprint(p, target):
            counts.filtered_own_preprint += 1
        elif is_future(p, target):
            counts.filtered_future += 1
            logger.warning(
                "index returned %s (dated %s) past the cutoff %s",
                p.key,
                p.published_date or p.year,
                target.cutoff,
            )
        elif target.cutoff is not None and is_undated(p, target):
            counts.filtered_undated += 1
        else:
            out.append(p)
    return out


def baseline_queries(query: NoveltyQuery) -> list[SearchQuery]:
    """The v0 queries: the statement, and method + problem."""
    out = [SearchQuery(id="b0", origin="baseline", kind="statement", text=query.statement)]
    mp = method_problem_query(query)
    if mp:
        out.append(SearchQuery(id="b1", origin="baseline", kind="method_problem", text=mp))
    return out


def _source(q: SearchQuery) -> CandidateSource:
    if q.origin != "baseline":
        return CandidateSource.generated
    if q.kind == "statement":
        return CandidateSource.search_statement
    return CandidateSource.search_method_problem


def _add(
    pool: dict[str, Candidate],
    paper: IndexPaper,
    source: CandidateSource,
    via: str | None = None,
    hit: QueryHit | None = None,
) -> bool:
    """Merge `paper` into `pool`; True if it is new."""
    sim = paper.similarity if paper.similarity is not None else 0.0
    existing = pool.get(paper.key)
    if existing is None:
        pool[paper.key] = Candidate(
            paper=paper,
            similarity=sim,
            sources=[source],
            via=[via] if via else [],
            hits=[hit] if hit else [],
        )
        return True
    if source not in existing.sources:
        existing.sources.append(source)
    if via and via not in existing.via:
        existing.via.append(via)
    if hit:
        existing.hits.append(hit)
    existing.similarity = max(existing.similarity, sim)
    if existing.paper.abstract is None and paper.abstract:
        existing.paper = paper.model_copy(update={"similarity": existing.paper.similarity})
    return False


def by_similarity(cands: list[Candidate]) -> list[Candidate]:
    return sorted(cands, key=lambda c: (-c.similarity, c.paper.key))


def interleave(lanes: list[list[str]], limit: int, titles: dict[str, str]) -> list[str]:
    """Round-robin by rank over the lanes (every lane's 1st key, then every 2nd, ...), skipping
    keys already taken and duplicates of a taken (long enough) title, up to `limit` keys. This
    is the first-pass pool filter: each query's top hits get in even when only one query finds
    them."""
    out: list[str] = []
    taken: set[str] = set()
    seen_titles: set[str] = set()
    depth = max((len(lane) for lane in lanes), default=0)
    for r in range(depth):
        for lane in lanes:
            if len(out) >= limit:
                return out
            if r >= len(lane) or lane[r] in taken:
                continue
            key = lane[r]
            taken.add(key)
            t = titles.get(key, "")
            if len(t.split()) >= MIN_TITLE_WORDS_FOR_MATCH:
                if t in seen_titles:
                    continue
                seen_titles.add(t)
            out.append(key)
    return out


def run_searches(
    queries: list[SearchQuery], index: IndexAPI, target: Target, params: RetrievalParams
) -> tuple[dict[str, Candidate], list[list[str]], CandidateCounts]:
    """Every query against /search (cutoff as `before`): (candidates by key, per-query lanes of
    keys in rank order, counts)."""
    counts = CandidateCounts(n_queries=len(queries))
    pool: dict[str, Candidate] = {}
    lanes: list[list[str]] = []
    for q in queries:
        k = params.search_k if q.origin == "baseline" else params.query_k
        hits = _admit(index.search(q.text, k, target.cutoff), target, counts)
        if q.kind == "statement":
            counts.search_statement = len(hits)
        elif q.kind == "method_problem":
            counts.search_method_problem = len(hits)
        for rank, p in enumerate(hits, start=1):
            _add(
                pool,
                p,
                _source(q),
                hit=QueryHit(query_id=q.id, rank=rank, similarity=p.similarity),
            )
        lanes.append([p.key for p in hits])
    counts.search_union = len(pool)
    counts.search_generated = sum(CandidateSource.generated in c.sources for c in pool.values())
    return pool, lanes, counts


def retrieve_multi(
    queries: list[SearchQuery],
    index: IndexAPI,
    target: Target,
    params: RetrievalParams,
    expansion_text: str,
) -> tuple[list[Candidate], CandidateCounts, list[Candidate]]:
    """Run every query (with the cutoff as `before`), expand one hop, and fill the LLM pool
    round-robin across queries. Returns (pool in fill order, counts, retrieved-but-unpooled)."""
    pool, lanes, counts = run_searches(queries, index, target, params)
    before = target.cutoff
    titles = {k: normalize_title(c.paper.title) for k, c in pool.items()}

    seeds = interleave(lanes, params.expand_top, titles)
    counts.expansion_nodes = len(seeds)
    expansion_lane: list[IndexPaper] = []
    for seed in seeds:
        neigh = index.neighbors(seed, expansion_text, before)
        if not neigh:
            counts.expansion_nodes_empty += 1
        admitted = _admit(neigh, target, counts)
        admitted.sort(key=lambda p: -(p.similarity or 0.0))
        for p in admitted[: params.neighbors_per_node]:
            if _add(pool, p, CandidateSource.expansion, via=seed):
                counts.expansion_new += 1
                expansion_lane.append(p)
    expansion_lane.sort(key=lambda p: (-(p.similarity or 0.0), p.key))
    titles.update({p.key: normalize_title(p.title) for p in expansion_lane})

    order = interleave([*lanes, [p.key for p in expansion_lane]], params.pool_size, titles)
    chosen = [pool[k] for k in order]
    in_pool = set(order)
    unpooled = by_similarity([c for k, c in pool.items() if k not in in_pool])
    counts.pool = len(chosen)
    counts.pool_from_expansion = sum(c.sources == [CandidateSource.expansion] for c in chosen)
    counts.pool_from_generated_only = sum(c.sources == [CandidateSource.generated] for c in chosen)
    logger.info(
        "retrieval: %d queries -> %d union (%d from generated queries), +%d from expansion "
        "(%d/%d seeds had no neighbors); pool %d (%d generated-only, %d expansion-only); "
        "dropped %d self, %d own preprint, %d without title/abstract; guard %d past cutoff, "
        "%d undated",
        counts.n_queries,
        counts.search_union,
        counts.search_generated,
        counts.expansion_new,
        counts.expansion_nodes_empty,
        counts.expansion_nodes,
        counts.pool,
        counts.pool_from_generated_only,
        counts.pool_from_expansion,
        counts.filtered_self,
        counts.filtered_own_preprint,
        counts.filtered_no_metadata,
        counts.filtered_future,
        counts.filtered_undated,
    )
    return chosen, counts, unpooled


def retrieve(
    query: NoveltyQuery, index: IndexAPI, target: Target, params: RetrievalParams
) -> tuple[list[Candidate], CandidateCounts]:
    """v0 retrieval: the two baseline queries only."""
    pool, counts, _ = retrieve_multi(
        baseline_queries(query), index, target, params, query.statement
    )
    return pool, counts
