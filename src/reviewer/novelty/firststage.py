"""First-stage ranker (no LLM): fill the rerank pool from search + /expand candidates.

Candidates are the union of every query's search hits and the one-hop citation neighbors of the
top search results (POST /expand). Each gets a small feature vector; a hand-set rule or a
logistic regression scores it and the top `pool_size` go to the LLM. The paper's own reference
list is never used (no `target` is sent to /expand, and no feature depends on it).
"""

import json
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from reviewer.models import (
    Candidate,
    CandidateCounts,
    CandidateSource,
    ExpandCandidate,
    IndexPaper,
    SearchQuery,
)
from reviewer.novelty.index_client import IndexAPI
from reviewer.novelty.retrieve import (
    RetrievalParams,
    Target,
    _add,
    _admit,
    by_similarity,
    interleave,
    normalize_title,
    run_searches,
)

logger = logging.getLogger(__name__)

FEATURES = (
    "max_cosine",
    "log_seed_count",
    "inv_best_rank",
    "in_search",
    "in_expansion",
    "in_both",
)
MAX_EXPAND_QUERIES = 20


@dataclass
class StageCandidate:
    candidate: Candidate
    max_cosine: float
    """Max cosine to any generated query (search hits of generated queries, or /expand's
    max_cosine); falls back to the best similarity to any query when none is known."""
    seed_count: int
    best_rank: int | None
    """Best rank over all search queries (None: only reached by expansion)."""
    in_search: bool
    in_expansion: bool

    def features(self) -> list[float]:
        return [
            self.max_cosine,
            math.log1p(self.seed_count),
            1.0 / self.best_rank if self.best_rank else 0.0,
            float(self.in_search),
            float(self.in_expansion),
            float(self.in_search and self.in_expansion),
        ]


Scorer = Callable[[np.ndarray], np.ndarray]
"""Feature matrix (n x len(FEATURES)) -> scores (higher = more likely prior work)."""


def rule_score(x: np.ndarray) -> np.ndarray:
    """Hand-set rule: semantic closeness first, plus bonuses for being co-cited by several seeds,
    for a top search rank, and for being found both ways."""
    return x[:, 0] + 0.03 * x[:, 1] + 0.05 * x[:, 2] + 0.02 * x[:, 5]


@dataclass
class LogisticModel:
    weights: list[float]
    bias: float
    mean: list[float]
    std: list[float]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        z = (x - np.array(self.mean)) / np.array(self.std)
        return z @ np.array(self.weights) + self.bias

    def to_json(self) -> str:
        return json.dumps({"features": list(FEATURES), **self.__dict__}, indent=2)

    @classmethod
    def load(cls, path: Path) -> "LogisticModel":
        d = json.loads(path.read_text(encoding="utf-8"))
        if tuple(d["features"]) != FEATURES:
            raise ValueError(f"{path}: trained on features {d['features']}, expected {FEATURES}")
        return cls(weights=d["weights"], bias=d["bias"], mean=d["mean"], std=d["std"])


def fit_logistic(
    x: np.ndarray, y: np.ndarray, l2: float = 1.0, iterations: int = 50
) -> LogisticModel:
    """L2-regularised logistic regression (Newton's method) on standardised features, with
    positives up-weighted to balance the classes (one named work among ~1,000 candidates)."""
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    z = np.hstack([(x - mean) / std, np.ones((len(x), 1))])
    n_pos = max(float(y.sum()), 1.0)
    sample_w = np.where(y == 1, (len(y) - n_pos) / n_pos, 1.0)
    w = np.zeros(z.shape[1])
    reg = np.full(z.shape[1], l2)
    reg[-1] = 0.0  # no penalty on the bias
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-(z @ w)))
        grad = z.T @ (sample_w * (p - y)) + reg * w
        hess = (z * (sample_w * p * (1 - p))[:, None]).T @ z + np.diag(reg)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return LogisticModel(
        weights=w[:-1].tolist(), bias=float(w[-1]), mean=mean.tolist(), std=std.tolist()
    )


def fit_folds(
    tables: Sequence[np.ndarray], labels: Sequence[np.ndarray], folds: int = 5
) -> list[LogisticModel]:
    """One model per fold, trained on every case not in that fold (case i is in fold i % folds;
    folds are groups of whole cases)."""
    models = []
    for f in range(folds):
        train = [i for i in range(len(tables)) if i % folds != f and len(tables[i])]
        x = np.vstack([tables[i] for i in train])
        y = np.concatenate([labels[i] for i in train])
        models.append(fit_logistic(x, y))
    return models


def cross_validated_scores(
    tables: Sequence[np.ndarray],
    labels: Sequence[np.ndarray],
    folds: int = 5,
    models: list[LogisticModel] | None = None,
) -> list[np.ndarray]:
    """Scores for each case's candidates from the model that did not see it."""
    models = models or fit_folds(tables, labels, folds)
    return [models[i % len(models)](t) if len(t) else np.zeros(0) for i, t in enumerate(tables)]


@dataclass
class CrossValidatedScorer:
    """Fold models plus each dev case's fold, so an eval run scores every case with the model
    trained without it (never the model fitted on all cases)."""

    models: list[LogisticModel]
    fold_of: dict[str, int]

    def for_case(self, case_id: str) -> LogisticModel:
        if case_id not in self.fold_of:
            raise ValueError(f"case {case_id} was not in the first-stage CV set")
        return self.models[self.fold_of[case_id]]

    def to_json(self) -> str:
        return json.dumps(
            {
                "features": list(FEATURES),
                "fold_of": self.fold_of,
                "models": [json.loads(m.to_json()) for m in self.models],
            },
            indent=2,
        )

    @classmethod
    def load(cls, path: Path) -> "CrossValidatedScorer":
        d = json.loads(path.read_text(encoding="utf-8"))
        if tuple(d["features"]) != FEATURES:
            raise ValueError(f"{path}: trained on features {d['features']}, expected {FEATURES}")
        models = [
            LogisticModel(weights=m["weights"], bias=m["bias"], mean=m["mean"], std=m["std"])
            for m in d["models"]
        ]
        return cls(models=models, fold_of=d["fold_of"])


def select_pool(cands: list[StageCandidate], scores: np.ndarray, size: int) -> list[int]:
    """Indices of the top `size` candidates by score (ties: similarity, then key), skipping
    duplicates of a (long enough) title already taken."""
    order = sorted(
        range(len(cands)),
        key=lambda i: (-scores[i], -cands[i].candidate.similarity, cands[i].candidate.paper.key),
    )
    out: list[int] = []
    seen_titles: set[str] = set()
    for i in order:
        if len(out) >= size:
            break
        t = normalize_title(cands[i].candidate.paper.title)
        if len(t.split()) >= 4:
            if t in seen_titles:
                continue
            seen_titles.add(t)
        out.append(i)
    return out


def _expand_paper(c: ExpandCandidate) -> IndexPaper:
    return IndexPaper(
        key=c.key,
        kind=c.kind,
        title=c.title,
        abstract=c.abstract,
        year=c.year,
        published_date=c.published_date,
        similarity=c.max_cosine,
    )


def gather_candidates(
    queries: list[SearchQuery], index: IndexAPI, target: Target, params: RetrievalParams
) -> tuple[list[StageCandidate], CandidateCounts]:
    """Search with every query, expand the top `expand_seeds` search results one hop through
    POST /expand (scored against the generated queries), and build each candidate's features."""
    pool, lanes, counts = run_searches(queries, index, target, params)
    titles = {k: normalize_title(c.paper.title) for k, c in pool.items()}
    seeds = interleave(lanes, params.expand_seeds, titles)
    generated = {q.id for q in queries if q.origin != "baseline"}
    expand_texts = [q.text for q in queries if q.origin != "baseline"][:MAX_EXPAND_QUERIES]
    if not expand_texts:
        expand_texts = [q.text for q in queries][:MAX_EXPAND_QUERIES]

    expanded: dict[str, tuple[int, float | None]] = {}
    counts.expansion_nodes = len(seeds)
    if seeds:
        resp = index.expand(seeds, expand_texts, target.cutoff, params.expand_max_candidates)
        papers = [_expand_paper(c) for c in resp.candidates]
        admitted = {p.key for p in _admit(papers, target, counts)}
        for c, p in zip(resp.candidates, papers, strict=True):
            if p.key not in admitted:
                continue
            expanded[c.key] = (c.seed_count, c.max_cosine)
            via = [link.seed for link in c.seeds]
            is_new = _add(pool, p, CandidateSource.expansion, via=via[0] if via else None)
            for extra in via[1:]:
                if extra not in pool[c.key].via:
                    pool[c.key].via.append(extra)
            counts.expansion_new += is_new

    out: list[StageCandidate] = []
    for key, cand in pool.items():
        gen_sims = [h.similarity for h in cand.hits if h.query_id in generated and h.similarity]
        seed_count, exp_cos = expanded.get(key, (0, None))
        cos_values = [*gen_sims, *([exp_cos] if exp_cos is not None else [])]
        max_cos = max(cos_values) if cos_values else cand.similarity
        out.append(
            StageCandidate(
                candidate=cand,
                max_cosine=max_cos,
                seed_count=seed_count,
                best_rank=min((h.rank for h in cand.hits), default=None),
                in_search=bool(cand.hits),
                in_expansion=key in expanded,
            )
        )
    return out, counts


def feature_matrix(cands: list[StageCandidate]) -> np.ndarray:
    if not cands:
        return np.zeros((0, len(FEATURES)))
    return np.array([c.features() for c in cands], dtype=float)


def retrieve_first_stage(
    queries: list[SearchQuery],
    index: IndexAPI,
    target: Target,
    params: RetrievalParams,
    scorer: Scorer,
) -> tuple[list[Candidate], CandidateCounts, list[Candidate]]:
    """(pool in score order, counts, retrieved-but-unpooled) using the first-stage `scorer`."""
    cands, counts = gather_candidates(queries, index, target, params)
    scores = scorer(feature_matrix(cands)) if cands else np.zeros(0)
    chosen_idx = select_pool(cands, scores, params.pool_size)
    chosen = [cands[i].candidate for i in chosen_idx]
    taken = set(chosen_idx)
    unpooled = by_similarity([c.candidate for i, c in enumerate(cands) if i not in taken])
    counts.pool = len(chosen)
    counts.pool_from_expansion = sum(c.sources == [CandidateSource.expansion] for c in chosen)
    counts.pool_from_generated_only = sum(c.sources == [CandidateSource.generated] for c in chosen)
    logger.info(
        "first stage: %d candidates (%d from search, %d new via /expand from %d seeds); "
        "pool %d (%d expansion-only)",
        len(cands),
        counts.search_union,
        counts.expansion_new,
        counts.expansion_nodes,
        counts.pool,
        counts.pool_from_expansion,
    )
    return chosen, counts, unpooled
