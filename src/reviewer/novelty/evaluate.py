"""Recall of reviewer-named prior work on data/novelty_eval/cases.jsonl (reviewer-index)."""

import logging
import re
from datetime import date
from pathlib import Path

from reviewer.cache import content_key
from reviewer.models import (
    Candidate,
    CandidateSource,
    ClaimNovelty,
    IndexCaseRow,
    IndexPaper,
    NamedJudgment,
    NoveltyEvalCase,
    NoveltyEvalCaseResult,
    NoveltyLabel,
    NoveltyQuery,
    PriorWorkRef,
    SearchQuery,
)
from reviewer.novelty.retrieve import MIN_TITLE_WORDS_FOR_MATCH, Target, normalize_title

logger = logging.getLogger(__name__)


VENUE_DEADLINES: dict[str, date] = {
    "ICLR.cc/2024/Conference": date(2023, 9, 28),
    "ICLR.cc/2025/Conference": date(2024, 10, 1),
}
"""Submission deadlines: the cutoff for prior work of a submission to that venue."""
_VENUE_YEAR = re.compile(r"/(\d{4})/")


def venue_cutoff(venue: str) -> date | None:
    """Known deadline, else Dec 31 of the year before the venue year, else no cutoff."""
    if venue in VENUE_DEADLINES:
        return VENUE_DEADLINES[venue]
    m = _VENUE_YEAR.search(venue)
    if not m:
        logger.warning("no cutoff for venue %r", venue)
        return None
    logger.warning("no deadline for venue %r; using end of the previous year", venue)
    return date(int(m.group(1)) - 1, 12, 31)


def group_cases(rows: list[IndexCaseRow]) -> list[NoveltyEvalCase]:
    """One case per submission with >= 1 resolved named work from before its cutoff; distinct
    works by index key."""
    by_sub: dict[str, list[IndexCaseRow]] = {}
    for r in rows:
        by_sub.setdefault(r.submission_id, []).append(r)
    cases = []
    for sub_id, sub_rows in by_sub.items():
        refs: dict[str, PriorWorkRef] = {}
        unresolved: set[str] = set()
        after_cutoff: set[str] = set()
        for r in sub_rows:
            res = r.resolution
            if res.status == "resolved" and res.index_key and res.after_cutoff:
                after_cutoff.add(res.index_key)
            elif res.status == "resolved" and res.index_key:
                refs.setdefault(
                    res.index_key, PriorWorkRef(key=res.index_key, title=res.matched_title)
                )
            else:
                unresolved.add(normalize_title(r.named_prior_work.title) or r.case_id)
        if not refs:
            continue
        first = sub_rows[0]
        cases.append(
            NoveltyEvalCase(
                case_id=sub_id,
                venue=first.venue,
                title=first.submission_title,
                abstract=first.submission_abstract,
                cutoff_date=first.cutoff_date or venue_cutoff(first.venue),
                prior_work=list(refs.values()),
                n_unresolved=len(unresolved),
                n_after_cutoff=len(after_cutoff - refs.keys()),
            )
        )
    return cases


def load_cases(path: Path) -> list[NoveltyEvalCase]:
    rows = [
        IndexCaseRow.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cases = group_cases(rows)
    logger.info("eval: %d rows -> %d submissions with resolved named work", len(rows), len(cases))
    return cases


def sample_cases(cases: list[NoveltyEvalCase], limit: int | None) -> list[NoveltyEvalCase]:
    """A deterministic, order-independent sample: cases sorted by a hash of their id."""
    ordered = sorted(cases, key=lambda c: content_key("novelty-eval-sample", c.case_id))
    return ordered[:limit] if limit is not None else ordered


def case_query(case: NoveltyEvalCase) -> NoveltyQuery:
    """No claims are extracted for eval submissions: the statement is title + abstract and the
    second query is the title."""
    statement = " ".join(p for p in (case.title, case.abstract) if p).strip()
    return NoveltyQuery(statement=statement, problem=case.title)


def case_target(case: NoveltyEvalCase) -> Target:
    return Target(title=case.title or None, abstract=case.abstract, cutoff=case.cutoff_date)


def _norm_key(key: str) -> str:
    return key.strip().lower().removeprefix("arxiv:")


def matches(ref: PriorWorkRef, paper: IndexPaper) -> bool:
    if ref.key and _norm_key(ref.key) == _norm_key(paper.key):
        return True
    t = normalize_title(ref.title)
    return len(t.split()) >= MIN_TITLE_WORDS_FOR_MATCH and t == normalize_title(paper.title)


def first_rank(ref: PriorWorkRef, papers: list[IndexPaper]) -> int | None:
    return next((i for i, p in enumerate(papers, start=1) if matches(ref, p)), None)


def recall_at(found_at: list[int | None], k: int) -> float:
    if not found_at:
        return 0.0
    return sum(r is not None and r <= k for r in found_at) / len(found_at)


def _provenance(c: Candidate) -> list[str]:
    ids = sorted({h.query_id for h in c.hits})
    return ids + (["x"] if CandidateSource.expansion in c.sources else [])


def score_case(case: NoveltyEvalCase, result: ClaimNovelty) -> NoveltyEvalCaseResult:
    """Ranks of the named works in the judged list, whether they reached the pool (judged) or
    only the retrieved union, and which queries retrieved them."""
    ranked = [j.candidate.paper for j in result.ranked]
    retrieved = [j.candidate for j in result.ranked] + result.unpooled
    named_judgments: list[NamedJudgment | None] = []
    for ref in case.prior_work:
        pos = first_rank(ref, ranked)
        if pos is None:
            named_judgments.append(None)
            continue
        j = result.ranked[pos - 1]
        named_judgments.append(
            NamedJudgment(
                key=j.candidate.paper.key,
                title=j.candidate.paper.title,
                abstract=j.candidate.paper.abstract,
                label=j.label,
                reason=j.reason,
                rank=pos,
                listwise_rank=j.listwise_rank,
            )
        )
    found_at = [first_rank(ref, ranked) for ref in case.prior_work]
    in_pool = [r is not None for r in found_at]
    found_by: list[list[str]] = []
    for ref in case.prior_work:
        cand = next((c for c in retrieved if matches(ref, c.paper)), None)
        found_by.append(_provenance(cand) if cand else [])
    in_union = [bool(f) for f in found_by]
    n = len(case.prior_work)
    return NoveltyEvalCaseResult(
        case_id=case.case_id,
        venue=case.venue,
        n_named=n,
        n_unresolved=case.n_unresolved,
        n_after_cutoff=case.n_after_cutoff,
        found_at=found_at,
        in_pool=in_pool,
        in_union=in_union,
        found_by=found_by,
        queries=result.queries,
        named_judgments=named_judgments,
        severity=result.severity,
        recall_at_5=recall_at(found_at, 5),
        recall_at_10=recall_at(found_at, 10),
        recall_at_20=recall_at(found_at, 20),
        pool_recall=sum(in_pool) / n if n else 0.0,
        union_recall=sum(in_union) / n if n else 0.0,
        verdict=result.verdict,
    )


def score_retrieval(
    case: NoveltyEvalCase,
    queries: list[SearchQuery],
    pool: list[Candidate],
    unpooled: list[Candidate],
) -> NoveltyEvalCaseResult:
    """Retrieval-only scoring: union and pool recall (no ranking, so no recall@k)."""
    pooled_papers = [c.paper for c in pool]
    retrieved = pool + unpooled
    in_pool = [first_rank(ref, pooled_papers) is not None for ref in case.prior_work]
    found_by = []
    for ref in case.prior_work:
        cand = next((c for c in retrieved if matches(ref, c.paper)), None)
        found_by.append(_provenance(cand) if cand else [])
    in_union = [bool(f) for f in found_by]
    n = len(case.prior_work)
    return NoveltyEvalCaseResult(
        case_id=case.case_id,
        venue=case.venue,
        n_named=n,
        n_unresolved=case.n_unresolved,
        n_after_cutoff=case.n_after_cutoff,
        found_at=[None] * n,
        in_pool=in_pool,
        in_union=in_union,
        found_by=found_by,
        queries=queries,
        recall_at_5=0.0,
        recall_at_20=0.0,
        pool_recall=sum(in_pool) / n if n else 0.0,
        union_recall=sum(in_union) / n if n else 0.0,
    )


def micro(results: list[NoveltyEvalCaseResult]) -> dict[str, float]:
    """Recall over all named works pooled across scored cases (failed cases excluded)."""
    ok = [r for r in results if r.error is None]
    named = sum(r.n_named for r in ok)
    if not named:
        return {
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "recall_at_20": 0.0,
            "pool_recall": 0.0,
            "union_recall": 0.0,
        }
    all_ranks = [rank for r in ok for rank in r.found_at]
    return {
        "recall_at_5": recall_at(all_ranks, 5),
        "recall_at_10": recall_at(all_ranks, 10),
        "recall_at_20": recall_at(all_ranks, 20),
        "pool_recall": sum(sum(r.in_pool) for r in ok) / named,
        "union_recall": sum(sum(r.in_union) for r in ok) / named,
    }


def query_group(q: SearchQuery | None, qid: str) -> str:
    """Reporting bucket for a query: baseline:statement, claim:technique, paper:objective, ...;
    'expansion' for 'x'."""
    if qid == "x":
        return "expansion"
    if q is None:
        return f"unknown:{qid}"
    return f"{q.origin}:{q.kind}"


def by_query_group(results: list[NoveltyEvalCaseResult]) -> dict[str, dict[str, int]]:
    """Per query group: named works it retrieved ('found'), those no other group retrieved
    ('only'), and how many of its finds reached the pool and the top 5."""
    out: dict[str, dict[str, int]] = {}
    for r in results:
        if r.error is not None:
            continue
        qmap = {q.id: q for q in r.queries}
        for ids, rank, pooled in zip(r.found_by, r.found_at, r.in_pool, strict=True):
            groups = {query_group(qmap.get(i), i) for i in ids}
            for g in groups:
                row = out.setdefault(g, {"found": 0, "only": 0, "pooled": 0, "top5": 0})
                row["found"] += 1
                row["only"] += len(groups) == 1
                row["pooled"] += pooled
                row["top5"] += rank is not None and rank <= 5
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["found"]))


def label_counts(results: list[NoveltyEvalCaseResult]) -> dict[str, int]:
    """Labels the reranker gave the named works that reached the pool."""
    counts = {label.value: 0 for label in NoveltyLabel}
    for r in results:
        for nj in r.named_judgments:
            if nj is not None:
                counts[nj.label.value] += 1
    return counts


def different_examples(
    results: list[NoveltyEvalCaseResult], n: int = 10
) -> list[tuple[str, str, NamedJudgment]]:
    """(case id, submission statement, judgment) for named works labelled `different`, in the
    (hash-sorted) case order."""
    out = []
    for r in results:
        statement = next((q.text for q in r.queries if q.id == "b0"), "")
        for nj in r.named_judgments:
            if nj is not None and nj.label == NoveltyLabel.different:
                out.append((r.case_id, statement, nj))
    return out[:n]


def auc(positives: list[float], negatives: list[float]) -> float:
    """P(random positive scores above random negative), ties counting half (Mann-Whitney)."""
    if not positives or not negatives:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))
