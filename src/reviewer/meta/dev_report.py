"""The 60-paper dev run report (plan §10): correlation with mean reviewer rating, decision AUC,
rating distributions by decision group, the accepted-papers-at-the-floor flag, and the
contamination-probe split. Pure function of `DevRow`s; no LLM, no network."""

import random
from collections.abc import Callable

from reviewer.meta.metrics import by_group_distribution, decision_auc, score_histogram, spearman
from reviewer.models import DevRow

NOT_KNOWN_CONFIDENCE_MAX = 2
"""Contamination probe confidence at or below this: treated as "doesn't seem to know this
paper" for the split report."""


def is_accepted(row: DevRow) -> bool:
    return row.final_status == "accepted"


def _spearman_of(rows: list[DevRow], getter: Callable[[DevRow], float | None]) -> float:
    pairs = [(getter(r), r.mean_reviewer_rating) for r in rows if getter(r) is not None]
    if len(pairs) < 2:
        return float("nan")
    xs, ys = zip(*pairs, strict=True)
    return spearman(list(xs), list(ys))


def correlation_block(rows: list[DevRow]) -> dict[str, float]:
    return {
        "meta_raw": _spearman_of(rows, lambda r: r.meta_raw_rating),
        "baseline_abstract": _spearman_of(rows, lambda r: r.baseline_abstract_rating),
        "baseline_full": _spearman_of(rows, lambda r: r.baseline_full_rating),
    }


def _auc_of(rows: list[DevRow], getter: Callable[[DevRow], float | None]) -> float:
    pos = [getter(r) for r in rows if is_accepted(r) and getter(r) is not None]
    neg = [getter(r) for r in rows if not is_accepted(r) and getter(r) is not None]
    return decision_auc(pos, neg)  # type: ignore[arg-type]


def auc_block(rows: list[DevRow]) -> dict[str, float]:
    """Accepted vs. not-accepted (rejected and withdrawn are both treated as the negative
    class -- a withdrawn paper is not evidence the venue would have accepted it)."""
    return {
        "meta_raw": _auc_of(rows, lambda r: r.meta_raw_rating),
        "baseline_abstract": _auc_of(rows, lambda r: r.baseline_abstract_rating),
        "baseline_full": _auc_of(rows, lambda r: r.baseline_full_rating),
    }


def floor_flag(rows: list[DevRow], floor: float = 1.0) -> dict[str, object]:
    """Accepted papers whose meta raw rating hit the venue's floor: does the checker-driven
    review score real accepted papers at the bottom too, the way both smoke-test papers did?"""
    accepted_rows = [r for r in rows if is_accepted(r)]
    at_floor = [r for r in accepted_rows if r.meta_raw_rating <= floor]
    return {
        "n_accepted": len(accepted_rows),
        "n_accepted_at_floor": len(at_floor),
        "accepted_at_floor_paper_ids": [r.paper_id for r in at_floor],
    }


def contamination_split(rows: list[DevRow]) -> dict[str, object]:
    known = [r for r in rows if (r.contamination_confidence or 0) > NOT_KNOWN_CONFIDENCE_MAX]
    unknown = [r for r in rows if (r.contamination_confidence or 0) <= NOT_KNOWN_CONFIDENCE_MAX]

    def block(subset: list[DevRow]) -> dict[str, object]:
        return {
            "n": len(subset),
            "correlation": correlation_block(subset),
            "auc": auc_block(subset),
        }

    guessed = [r for r in rows if r.contamination_accepted_guess is not None]
    correct = sum(1 for r in guessed if r.contamination_accepted_guess == is_accepted(r))
    return {
        "known": block(known),
        "unknown": block(unknown),
        "probe_guess_accuracy": correct / len(guessed) if guessed else float("nan"),
        "n_guessed": len(guessed),
    }


def human_agreement(rows: list[DevRow], draws: int = 50, seed: int = 20260927) -> dict[str, float]:
    """The bar to compare against: one randomly chosen reviewer's rating vs the mean of the
    other reviewers (Spearman across papers), and that single reviewer's decision AUC; averaged
    over `draws` seeded draws. Papers with fewer than 2 reviews are skipped."""
    rng = random.Random(seed)
    usable = [r for r in rows if len(r.reviewer_ratings) >= 2]
    if len(usable) < 2:
        return {"n": len(usable), "spearman_one_vs_rest": float("nan"), "auc_one": float("nan")}
    sps, aucs = [], []
    for _ in range(draws):
        one, rest = [], []
        for r in usable:
            i = rng.randrange(len(r.reviewer_ratings))
            others = r.reviewer_ratings[:i] + r.reviewer_ratings[i + 1 :]
            one.append(float(r.reviewer_ratings[i]))
            rest.append(sum(others) / len(others))
        sps.append(spearman(one, rest))
        pos = [o for o, r in zip(one, usable, strict=True) if is_accepted(r)]
        neg = [o for o, r in zip(one, usable, strict=True) if not is_accepted(r)]
        aucs.append(decision_auc(pos, neg))
    return {
        "n": len(usable),
        "spearman_one_vs_rest": sum(sps) / len(sps),
        "auc_one": sum(aucs) / len(aucs),
    }


def build_report(rows: list[DevRow]) -> dict[str, object]:
    groups = [r.final_status for r in rows]
    return {
        "n_papers": len(rows),
        "n_by_group": {g: groups.count(g) for g in sorted(set(groups))},
        "correlation": correlation_block(rows),
        "auc": auc_block(rows),
        "human_agreement": human_agreement(rows),
        "rating_distribution_by_group": {
            "mean_reviewer_rating": by_group_distribution(
                [r.mean_reviewer_rating for r in rows], groups
            ),
            "meta_raw_rating": by_group_distribution([r.meta_raw_rating for r in rows], groups),
        },
        "meta_raw_rating_histogram_by_group": score_histogram(
            [round(r.meta_raw_rating) for r in rows], groups
        ),
        "accepted_floor_flag": floor_flag(rows),
        "contamination": contamination_split(rows),
    }
