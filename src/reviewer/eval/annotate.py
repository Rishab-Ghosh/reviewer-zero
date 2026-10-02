"""Fill the computed fields of PaperRecords, and derive the public view. Pure functions."""

from collections.abc import Sequence

from .checks import DEFAULT_DUPLICATE_THRESHOLD, build_idf, max_similarities, run_checks
from .gold_compare import compare_gold
from .schemas import PaperRecord, SimilarityHit


def annotate_records(
    records: Sequence[PaperRecord], threshold: float = DEFAULT_DUPLICATE_THRESHOLD
) -> list[PaperRecord]:
    """Copies of `records` with flags, per-claim max similarity and gold comparison filled in.

    The IDF is built over the extracted statements of exactly these records, so a report
    rendered from a subset (the public view) carries no statistics from the omitted papers.
    """
    statements = [c.statement for r in records if r.claims is not None for c in r.claims.claims]
    idf = build_idf(statements)
    out: list[PaperRecord] = []
    for r in records:
        update: dict[str, object] = {"flags": [], "similarities": [], "gold_comparison": None}
        if r.claims is not None:
            own = [c.statement for c in r.claims.claims]
            update["flags"] = run_checks(r.claims, r.parsed, idf, threshold)
            update["similarities"] = [
                SimilarityHit(claim_index=i, other_index=j, score=score)
                for i, (j, score) in enumerate(max_similarities(own, idf))
            ]
        if r.gold is not None:
            update["gold_comparison"] = compare_gold(r.gold, r.claims, idf)
        out.append(r.model_copy(update=update))
    return out


def public_records(records: Sequence[PaperRecord]) -> list[PaperRecord]:
    """Records safe to commit: no own-tier papers, no private gold (treated as no gold).

    Computed fields are cleared; run `annotate_records` on the result.
    """
    return [
        r.model_copy(
            update={
                "gold": None if r.gold_private else r.gold,
                "gold_private": False,
                "flags": [],
                "similarities": [],
                "gold_comparison": None,
            }
        )
        for r in records
        if not r.is_own
    ]
