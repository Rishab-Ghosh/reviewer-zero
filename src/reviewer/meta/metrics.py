"""Pure statistics for the dev run report (plan §4, §7): Spearman correlation, decision AUC
(reused from the novelty eval), and rating distributions by decision group. No LLM, no network."""

from collections import Counter

import numpy as np

from reviewer.novelty.evaluate import auc as decision_auc  # noqa: F401  (re-exported)


def rankdata(values: list[float]) -> list[float]:
    """Average ranks (1-based), ties sharing the mean rank of their positions."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation. NaN if either side has zero variance or fewer than 2 points."""
    if len(xs) != len(ys):
        raise ValueError(f"length mismatch: {len(xs)} vs {len(ys)}")
    if len(xs) < 2:
        return float("nan")
    rx, ry = np.array(rankdata(xs)), np.array(rankdata(ys))
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def rating_distribution(ratings: list[float]) -> dict[str, float]:
    """Count, mean, and quartiles of a list of ratings (empty-safe)."""
    if not ratings:
        return {"n": 0, "mean": float("nan"), "min": float("nan"), "max": float("nan")}
    arr = np.array(ratings)
    return {
        "n": len(ratings),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
    }


def by_group_distribution(values: list[float], groups: list[str]) -> dict[str, dict[str, float]]:
    """`rating_distribution` split by a parallel list of group labels."""
    if len(values) != len(groups):
        raise ValueError(f"length mismatch: {len(values)} vs {len(groups)}")
    by_group: dict[str, list[float]] = {}
    for v, g in zip(values, groups, strict=True):
        by_group.setdefault(g, []).append(v)
    return {g: rating_distribution(vs) for g, vs in sorted(by_group.items())}


def score_histogram(values: list[int], groups: list[str]) -> dict[str, dict[int, int]]:
    """Raw score counts (e.g. 1-10) per group, for spotting e.g. accepted papers scoring the
    floor."""
    if len(values) != len(groups):
        raise ValueError(f"length mismatch: {len(values)} vs {len(groups)}")
    by_group: dict[str, Counter[int]] = {}
    for v, g in zip(values, groups, strict=True):
        by_group.setdefault(g, Counter())[v] += 1
    return {g: dict(sorted(c.items())) for g, c in sorted(by_group.items())}
