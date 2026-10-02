"""The review_paper result: each checker's findings in a compact, evidence-first form.

Never included: any rating, score, sub-score or accept/reject verdict (MASTER_PLAN D11) or the
novelty verdict or severity score (D14). The meta-review is serialized only as the D12-filtered
`MetaReviewOut` allowlist (`meta_review_out`).
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from reviewer.models import ReviewStep, StepResult

if TYPE_CHECKING:
    from reviewer.config import Settings
    from reviewer.models import MetaReviewOut

logger = logging.getLogger(__name__)

DISPLAY_LABEL = {
    "same": "check first",
    "subsumes": "check first",
    "close": "close",
    "builds_on": "builds on",
    "different": "different",
}
METHODS_SHOWN = {"no", "unclear"}


def _load(results: list[StepResult], step: ReviewStep, work_dir: Path) -> dict[str, Any] | None:
    r = next((x for x in results if x.step == step), None)
    if r is None or r.output is None:
        return None
    return json.loads(Path(r.output).read_text(encoding="utf-8"))


def summarize(results: list[StepResult], work_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    claims = _load(results, ReviewStep.claims, work_dir)
    if claims:
        out["claims"] = [
            {"index": i, "statement": c["statement"]}
            for i, c in enumerate(claims.get("claims", []), 1)
        ]
    nov = _load(results, ReviewStep.novelty, work_dir)
    if nov:
        out["prior_work"] = []
        for c in nov.get("claims", []):
            sev = c.get("severity") or {}
            flag = None
            if sev.get("score") == 3:
                flag = {"work": (sev.get("drivers") or [None])[0], "reason": sev.get("reason")}
            out["prior_work"].append(
                {
                    "claim": (c.get("query") or {}).get("statement"),
                    "possible_prior_claim": flag,
                    "works": [
                        {
                            "key": j["candidate"]["paper"]["key"],
                            "title": j["candidate"]["paper"].get("title"),
                            "year": j["candidate"]["paper"].get("year"),
                            "label": DISPLAY_LABEL.get(j.get("label"), j.get("label")),
                            "reason": j.get("reason"),
                        }
                        for j in c.get("ranked", [])[:10]
                    ],
                }
            )
    meth = _load(results, ReviewStep.methods, work_dir)
    if meth:
        out["methodology"] = [
            {
                "item": f["item"],
                "verdict": f["verdict"],
                "agreement": f.get("agreement"),
                "concern": f.get("concern") or f.get("rationale"),
                "quotes": [q["quote"] for q in f.get("quotes", []) if q.get("verified")],
            }
            for f in meth.get("findings", [])
            if f.get("verdict") in METHODS_SHOWN
        ]
    wr = _load(results, ReviewStep.writing, work_dir)
    if wr:
        out["writing"] = [
            {
                "check": f["check"],
                "quote": f.get("quote"),
                "location": f.get("location"),
                "detail": f.get("detail"),
            }
            for f in wr.get("findings", [])
        ]
    cit = _load(results, ReviewStep.citations, work_dir)
    if cit:
        out["citations"] = {
            "flags": [
                {
                    "ref_id": f["ref_id"],
                    "type": f["type"],
                    "detail": f["detail"],
                    "reference": c["raw_string"][:300],
                }
                for c in cit.get("references", [])
                for f in c.get("flags", [])
            ],
            "possible_contradictions": [
                {
                    "ref_id": s["ref_id"],
                    "sentence": s["sentence"],
                    "quote": s.get("quote"),
                    "rationale": s.get("rationale"),
                }
                for s in cit.get("support", [])
                if s.get("verdict") == "contradicted"
            ],
        }
    return out


def meta_review_out(
    results: list[StepResult],
    settings: "Settings",
    venue: str,
    venue_year: int | None,
    failures: dict[str, str],
) -> "MetaReviewOut | None":
    """The meta step's judgment as the D12-filtered allowlist; None if the step has no output.
    Each removed sentence is recorded locally with its field and pattern (D12)."""
    from reviewer.failures import record_failure
    from reviewer.meta.calibration import display_rating
    from reviewer.meta.input_builder import build_meta_input
    from reviewer.meta.leak_filter import filter_meta
    from reviewer.models import MetaReview, ParsedPaper

    r = next((x for x in results if x.step == ReviewStep.meta), None)
    if r is None or r.output is None:
        return None
    review = MetaReview.model_validate_json(Path(r.output).read_text(encoding="utf-8"))
    if not review.judgments:
        return None
    j = review.judgments[0]
    parsed = settings.parsed_dir / f"{review.paper_id}.json"
    paper_text, evidence = "", []
    if parsed.exists():
        paper = ParsedPaper.model_validate_json(parsed.read_text(encoding="utf-8"))
        paper_text = " ".join(
            [paper.title, paper.abstract, *(" ".join(s.paragraphs) for s in paper.sections)]
        )
        try:
            mi = build_meta_input(
                settings, paper, venue=venue, venue_year=venue_year, failures=failures
            )
            evidence = [e.text for e in mi.evidence]
        except (OSError, ValueError) as e:
            logger.warning("meta evidence unavailable for the leak filter: %s", type(e).__name__)
    scores = [
        j.raw_rating,
        j.soundness.score,
        j.presentation.score,
        j.contribution.score,
        j.confidence,
        review.calibration.rating,
        display_rating(j.raw_rating),
    ]
    checks = [(c.checker.value, c.status.value == "ok") for c in review.checker_status]
    result = filter_meta(j, paper_text, evidence, scores, checks)
    for d in result.dropped:
        record_failure(
            settings.failures_file,
            "meta",
            review.paper_id,
            f"leak filter removed {d.field} ({d.pattern}): {d.sentence}",
        )
    return result.out
