"""Cost estimate before a review (MASTER_PLAN D5: "about $M", reserve = M × 1.3).

Per-step medians of measured fresh spend, from the three complete runs on file (2026-10-02):
2601.19871 and 2512.22388 (cached pipeline runs; their meta cost included the research
baselines D67 removes, so meta uses the demo's judgment-only call) and the fictional demo paper
(docs/demo). Update them as real runs accumulate.
"""

from reviewer.models import ReviewStep
from reviewer.review.budget import job_limit

STEP_MEDIAN_USD: dict[ReviewStep, float] = {
    ReviewStep.parse: 0.0,
    ReviewStep.format: 0.0,
    ReviewStep.claims: 0.091,  # 0.084, 0.117, 0.091
    ReviewStep.citations: 0.087,  # 0.087, 0.181, 0.081
    ReviewStep.methods: 0.168,  # 0.168, 0.248, 0.160
    ReviewStep.writing: 0.090,  # 0.090, 0.145, 0.074
    ReviewStep.novelty: 0.903,  # 0.903, 0.900, demo ≈ 1.18
    ReviewStep.meta: 0.120,  # demo, judgment only (k = 1)
}
NOVELTY_LITE_MEDIAN_USD = 0.38
"""M11 (2026-10-02): the novelty step of the live lite review_paper run on the demo paper,
fresh cache. One paper; the demo's full-pool novelty (≈ 1.18) is above the dev median, so this
is on the high side."""
RESERVE_FACTOR = 1.3
LARGEST_HOLD_USD = 0.5
"""Upper bound on one call's hold (an Opus meta/methods call on a long paper)."""


def estimate(remaining: list[ReviewStep], novelty: str = "full") -> tuple[float, float, float]:
    """(about, reserve, hard limit) in USD for the steps still to run."""
    medians = dict(STEP_MEDIAN_USD)
    if novelty == "lite":
        medians[ReviewStep.novelty] = NOVELTY_LITE_MEDIAN_USD
    about = round(sum(medians[s] for s in remaining), 2)
    reserve = round(about * RESERVE_FACTOR, 2)
    return about, reserve, round(job_limit(reserve, LARGEST_HOLD_USD), 2) if about else 0.0
