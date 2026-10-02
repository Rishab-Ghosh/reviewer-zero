"""Turn judgment(s) into a score (plan §4.2). Code only, fit on dev data -- never the LLM.

`calibrate()` returns the median raw rating, unscaled, with `fitted=False`: a feature-based fit
overfit at dev-set size and scored worse than the raw rating (STATUS, 2026-09-28).
"""

from reviewer.models import MetaCalibration, MetaJudgment, WeaknessSeverity

DISPLAY_INTERCEPT = 1.984
DISPLAY_SLOPE = 0.750
"""Display-only linear rescale of the raw rating: matches the mean (5.27) and SD (1.11) of
reviewer mean ratings on the 47 ICLR dev papers judged with meta_v1 (raw mean 4.38, SD 1.48).
Monotone, so Spearman and AUC are unchanged; evals use the stored raw rating, never this."""


def display_rating(raw: float) -> float:
    return min(10.0, max(1.0, DISPLAY_INTERCEPT + DISPLAY_SLOPE * raw))


def extract_features(judgments: list[MetaJudgment]) -> dict[str, float]:
    """The judgment features §4.2 names: sub-scores, raw rating, and weakness-severity counts.
    Median across runs when k > 1."""
    ratings = sorted(j.raw_rating for j in judgments)
    mid = len(ratings) // 2
    median_rating = ratings[mid] if len(ratings) % 2 else (ratings[mid - 1] + ratings[mid]) / 2
    n = len(judgments)
    return {
        "raw_rating": float(median_rating),
        "soundness": sum(j.soundness.score for j in judgments) / n,
        "presentation": sum(j.presentation.score for j in judgments) / n,
        "contribution": sum(j.contribution.score for j in judgments) / n,
        "confidence": sum(j.confidence for j in judgments) / n,
        "n_fatal": sum(
            w.severity == WeaknessSeverity.fatal for j in judgments for w in j.weaknesses
        )
        / n,
        "n_major": sum(
            w.severity == WeaknessSeverity.major for j in judgments for w in j.weaknesses
        )
        / n,
        "n_minor": sum(
            w.severity == WeaknessSeverity.minor for j in judgments for w in j.weaknesses
        )
        / n,
        "rating_spread": float(ratings[-1] - ratings[0]) if len(ratings) > 1 else 0.0,
    }


def calibrate(judgments: list[MetaJudgment]) -> MetaCalibration:
    """Placeholder calibration (plan §10 blocks a real fit until the 60-paper dev run exists):
    `rating` is the median raw judgment rating, unscaled; percentile/p_accept/tier are left
    unset rather than guessed. `fitted=False` marks this state."""
    if not judgments:
        return MetaCalibration(rating=0.0, fitted=False)
    features = extract_features(judgments)
    return MetaCalibration(
        rating=features["raw_rating"],
        rating_percentile=None,
        p_accept=None,
        tier="",
        fitted=False,
        features=features,
        weights={},
    )
