"""Render a MetaReview as markdown, OpenReview form order, plus its JSON. Pure."""

from reviewer.meta.calibration import display_rating
from reviewer.models import MetaReview, TokenUsage, Weakness, WeaknessSeverity

SEVERITY_ORDER = {WeaknessSeverity.fatal: 0, WeaknessSeverity.major: 1, WeaknessSeverity.minor: 2}


def _usage(u: TokenUsage) -> str:
    return f"{u.input_tokens:,} in / {u.output_tokens:,} out tokens, ${u.cost_usd:.2f}"


def _sorted_weaknesses(weaknesses: list[Weakness]) -> list[Weakness]:
    return sorted(weaknesses, key=lambda w: SEVERITY_ORDER[w.severity])


def render_checker_status(review: MetaReview) -> list[str]:
    lines = ["**Checkers used:**", ""]
    for s in review.checker_status:
        tag = {"ok": "ran", "failed": "**failed**", "not_run": "*not run*"}[s.status.value]
        lines.append(f"- {s.checker.value}: {tag}" + (f" ({s.detail})" if s.detail else ""))
    return lines


def render_review(review: MetaReview) -> str:
    j = review.judgments[0]
    cal = review.calibration
    lines = [
        f"# Meta-review: {review.paper_id}",
        "",
        f"*{review.title}*",
        "",
        f"Venue: {review.venue}{f' {review.venue_year}' if review.venue_year else ''}. "
        f"Model {review.model}, prompt {review.prompt_version}. Generated {review.generated}. "
        f"{_usage(review.token_usage)}.",
        "",
        *render_checker_status(review),
        "",
    ]
    if review.dropped_evidence_ids:
        lines += [
            f"*{review.dropped_evidence_ids} strength/weakness item(s) were dropped: their "
            "evidence ids did not resolve to a real finding.*",
            "",
        ]
    lines += [
        "## Summary",
        "",
        j.summary,
        "",
        "## Sub-scores",
        "",
        "| | score (1-4) | reason |",
        "|---|---|---|",
        f"| Soundness | {j.soundness.score} | {j.soundness.reason} |",
        f"| Presentation | {j.presentation.score} | {j.presentation.reason} |",
        f"| Contribution | {j.contribution.score} | {j.contribution.reason} |",
        "",
        "## Strengths",
        "",
    ]
    if not j.strengths:
        lines.append("None found.")
    for s in j.strengths:
        lines.append(
            f"- **[{s.category.value}]** {s.text} _(evidence: {', '.join(s.evidence_ids)})_"
        )
    lines += ["", "## Weaknesses", ""]
    if not j.weaknesses:
        lines.append("None found.")
    for w in _sorted_weaknesses(j.weaknesses):
        lines.append(
            f"- **[{w.severity.value}/{w.category.value}]** {w.text} "
            f"_(evidence: {', '.join(w.evidence_ids)})_"
        )
    lines += [
        "",
        "## Questions for the authors",
        "",
        *([f"- {q}" for q in j.questions] or ["None."]),
        "",
        "## What would change my mind",
        "",
        j.would_change_my_mind or "(not given)",
        "",
        "## Rating",
        "",
        f"Raw rating (this judgment run): **{j.raw_rating}**. Confidence: {j.confidence}/5.",
        "",
        f"**Display rating: {display_rating(cal.rating):.1f}** (the raw rating rescaled to the "
        "spread of ICLR reviewers' mean ratings; display only, fitted on 47 dev papers)",
        "",
    ]
    if len(review.judgments) > 1:
        ratings = [jj.raw_rating for jj in review.judgments]
        lines += [f"{len(review.judgments)} judgment runs; raw ratings: {ratings}.", ""]
    if review.baseline_abstract or review.baseline_full or review.contamination_probe:
        lines += ["## Baselines and contamination probe", ""]
        if review.baseline_abstract:
            b = review.baseline_abstract
            lines += [
                f"- **Abstract-only baseline (a):** raw rating **{b.raw_rating}** "
                f"(confidence {b.confidence}/5). {b.reasoning}",
            ]
        if review.baseline_full:
            b = review.baseline_full
            lines += [
                f"- **Full-text, no-checkers baseline (b):** raw rating **{b.raw_rating}** "
                f"(confidence {b.confidence}/5). {b.reasoning}",
            ]
        if review.contamination_probe:
            p = review.contamination_probe
            lines += [
                f"- **Contamination probe:** model guesses "
                f"{'accepted' if p.accepted_guess else 'not accepted'} "
                f"(confidence {p.confidence}/5). {p.reasoning}",
            ]
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
