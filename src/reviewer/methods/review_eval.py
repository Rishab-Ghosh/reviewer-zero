"""Evaluate the methodology checker against OpenReview reviewers' methodology concerns
(reviewer-index methods dataset). Dev only: the test split is locked in code.

Per dev submission: the checker reviews the PDF reviewers saw, then the existing Opus matcher
(match_v1, unchanged) matches every review concern to checker findings. Reported: recall per
concern category and overall (strict and lenient, full and with partial), per venue, whether the
matching finding came from the category's own checklist item, a lower bound on precision (the
share of strict findings some reviewer also raised), and the audited precision sample. Pure.
"""

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from reviewer.methods.evaluate import CritiqueOutcome, Recall, raised_concerns
from reviewer.models import (
    ChecklistItemId,
    ConcernOutcome,
    CritiqueEntry,
    CritiqueMatchLabel,
    Critiques,
    MethodsConcernRow,
    PaperMethodsReview,
)

logger = logging.getLogger(__name__)

CATEGORIES = (
    "baselines",
    "seeds_variance",
    "significance",
    "leakage_contamination",
    "compute_fairness",
    "ablations",
    "dataset_split_disclosure",
    "hyperparameter_disclosure",
    "claims_vs_results",
    "evaluation_scope",
    "other",
)
ALIASES = {
    "baseline": "baselines",
    "seeds": "seeds_variance",
    "variance": "seeds_variance",
    "statistical_significance": "significance",
    "leakage": "leakage_contamination",
    "contamination": "leakage_contamination",
    "compute": "compute_fairness",
    "compute_matched": "compute_fairness",
    "ablation": "ablations",
    "dataset_disclosure": "dataset_split_disclosure",
    "data_disclosure": "dataset_split_disclosure",
    "split_disclosure": "dataset_split_disclosure",
    "hparam_disclosure": "hyperparameter_disclosure",
    "hyperparameters": "hyperparameter_disclosure",
    "claims_vs_result": "claims_vs_results",
    "claims_results": "claims_vs_results",
    "scope": "evaluation_scope",
}
_I = ChecklistItemId
CATEGORY_ITEMS: dict[str, tuple[ChecklistItemId, ...]] = {
    "baselines": (_I.baseline_tuning,),
    "seeds_variance": (_I.seeds_variance,),
    "significance": (_I.significance,),
    "leakage_contamination": (_I.leakage,),
    "compute_fairness": (_I.compute_matched,),
    "ablations": (_I.ablation_isolation,),
    "dataset_split_disclosure": (_I.data_disclosure,),
    "hyperparameter_disclosure": (_I.hparam_disclosure,),
    "claims_vs_results": (_I.abstract_support, _I.claims_support),
    "evaluation_scope": (_I.evaluation_scope,),
    "other": (),
}
"""The checklist items (any version) that should raise each category; () = only
other_concerns can."""
_SLUG = re.compile(r"[^a-z0-9]+")

REVIEW_DEADLINES: dict[str, date] = {
    "ICLR.cc/2023/Conference": date(2022, 9, 28),
    "ICLR.cc/2024/Conference": date(2023, 9, 28),
    "ICLR.cc/2025/Conference": date(2024, 10, 1),
    "NeurIPS.cc/2023/Conference": date(2023, 5, 17),
    "NeurIPS.cc/2024/Conference": date(2024, 5, 22),
}
"""Full-paper deadlines, copied from reviewer-index novelty/split.py CUTOFFS (official CFP pages
cited there). Used only to pick the PDF revision reviewers saw; rows' own cutoff_date wins."""


class SplitLockedError(ValueError):
    """The methods-eval test split is locked: no code path may read it."""


def canonical_category(raw: str) -> str:
    slug = _SLUG.sub("_", raw.strip().lower()).strip("_")
    slug = ALIASES.get(slug, slug)
    if slug not in CATEGORIES:
        logger.warning("unknown concern category %r; kept as %r", raw, slug)
    return slug


def cases_path(eval_dir: Path, split: str, override: Path | None = None) -> Path:
    """dev.jsonl in the dataset dir (or `override`). Refuses anything that names test."""
    if split != "dev":
        raise SplitLockedError(f"split {split!r}: only dev is allowed; test is locked")
    path = override or eval_dir / "dev.jsonl"
    if "test" in path.name.lower():
        raise SplitLockedError(f"{path} looks like the test split; test is locked")
    return path


def load_rows(path: Path) -> list[MethodsConcernRow]:
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = MethodsConcernRow.model_validate_json(line)
        if (row.split or "dev").lower() == "test":
            raise SplitLockedError(f"{path}:{n} is marked split=test; test is locked")
        rows.append(row.model_copy(update={"category": canonical_category(row.category)}))
    return rows


@dataclass(frozen=True)
class Submission:
    submission_id: str
    venue: str
    title: str
    cutoff_date: date | None
    concerns: tuple[MethodsConcernRow, ...]
    final_status: str | None = None

    @property
    def paper_id(self) -> str:
        return f"or-{self.submission_id}"

    def critiques(self) -> Critiques:
        """The concerns in the matcher's input format (category kept as the source block;
        the matcher prompt does not see it)."""
        return Critiques(
            paper_id=self.paper_id,
            source_document=f"openreview:{self.submission_id}",
            entries=[
                CritiqueEntry(limitation=c.review_excerpt, source_block=c.category)
                for c in self.concerns
            ],
        )


def group_submissions(rows: list[MethodsConcernRow]) -> list[Submission]:
    """One submission per id, concerns in file order, duplicates (same review and excerpt)
    dropped; submissions sorted by id."""
    by: dict[str, list[MethodsConcernRow]] = defaultdict(list)
    for r in rows:
        if not any(
            x.review_id == r.review_id and x.review_excerpt == r.review_excerpt
            for x in by[r.submission_id]
        ):
            by[r.submission_id].append(r)
    out = []
    for sid in sorted(by):
        cs = by[sid]
        out.append(
            Submission(
                submission_id=sid,
                venue=cs[0].venue,
                title=cs[0].submission_title,
                cutoff_date=next((c.cutoff_date for c in cs if c.cutoff_date), None)
                or REVIEW_DEADLINES.get(cs[0].venue),
                final_status=cs[0].final_status,
                concerns=tuple(cs),
            )
        )
    return out


def concern_outcomes(
    sub: Submission,
    items: list[CritiqueOutcome],
    pdf_provenance: str = "",
    asked: set[str] | None = None,
) -> list[ConcernOutcome]:
    """`asked`: item ids the checker version asks; own_item is None when none of the category's
    items was asked."""
    out = []
    for o in items:
        row = sub.concerns[o.index]
        own = [i.value for i in CATEGORY_ITEMS.get(row.category, ()) if asked is None or i in asked]
        matched = o.lenient != CritiqueMatchLabel.none
        out.append(
            ConcernOutcome(
                submission_id=sub.submission_id,
                venue=sub.venue,
                review_id=row.review_id,
                category=row.category,
                excerpt=row.review_excerpt,
                scope=o.scope,
                strict=o.strict,
                lenient=o.lenient,
                finding_ids=list(o.finding_ids),
                own_item=any(i in o.finding_ids for i in own) if own and matched else None,
                concern_index=o.index,
                reason=o.reason,
                pdf_provenance=pdf_provenance,
                final_status=sub.final_status,
            )
        )
    return out


def recall_of(items: list[ConcernOutcome], mode: str) -> Recall:
    labels = [getattr(o, mode) for o in items]
    return Recall(
        n=len(labels),
        match=sum(x == CritiqueMatchLabel.match for x in labels),
        partial=sum(x == CritiqueMatchLabel.partial for x in labels),
    )


def by_key(items: list[ConcernOutcome], key: str) -> dict[str, list[ConcernOutcome]]:
    groups: dict[str, list[ConcernOutcome]] = defaultdict(list)
    for o in items:
        groups[str(getattr(o, key))].append(o)
    order = list(CATEGORIES) if key == "category" else sorted(groups)
    return {k: groups[k] for k in [*order, *sorted(set(groups) - set(order))] if k in groups}


def review_supported(
    reviews: list[PaperMethodsReview], items: list[ConcernOutcome]
) -> tuple[int, int]:
    """(strict findings some reviewer's concern matched or partly matched, strict findings).
    A lower bound on precision: reviews are not exhaustive."""
    cited = {(o.submission_id, k) for o in items if o.lenient != "none" for k in o.finding_ids}
    strict = [
        (r.paper_id.removeprefix("or-"), x.key)
        for r in reviews
        for x in raised_concerns(r)
        if x.strict
    ]
    return sum(s in cited for s in strict), len(strict)


def own_item_rate(items: list[ConcernOutcome]) -> tuple[int, int]:
    judged = [o.own_item for o in items if o.own_item is not None]
    return sum(judged), len(judged)


def _row(label: str, rc: Recall) -> str:
    return (
        f"| {label} | {rc.n} | {rc.match} | {rc.partial} | {rc.full:.0%} | {rc.with_partial:.0%} |"
    )


def render_review_eval(
    items: list[ConcernOutcome],
    reviews: list[PaperMethodsReview],
    precision_line: str,
    header: list[str],
) -> str:
    hdr = "| {} | n | match | partial | recall | incl. partial |"
    sep = "|---|---:|---:|---:|---:|---:|"
    lines = [*header, "", "## Overall", "", hdr.format("mode"), sep]
    for mode in ("strict", "lenient"):
        lines.append(_row(mode, recall_of(items, mode)))
    keys = (
        ("category", "category"),
        ("venue", "venue"),
        ("scope", "scope"),
        ("final_status", "decision"),
        ("pdf_provenance", "PDF provenance (current_revised: may already answer reviewers)"),
    )
    for key, title in keys:
        lines += ["", f"## By {title} (strict / lenient)", "", hdr.format(title), sep]
        for k, group in by_key(items, key).items():
            lines.append(_row(f"{k}, strict", recall_of(group, "strict")))
            lines.append(_row(f"{k}, lenient", recall_of(group, "lenient")))
    hit, n = own_item_rate(items)
    sup, n_strict = review_supported(reviews, items)
    lines += [
        "",
        "## Other diagnostics",
        "",
        f"- Matched concerns whose match includes the category's own checklist item: {hit}/{n}"
        + (f" ({hit / n:.0%})" if n else ""),
        f"- Strict checker findings that some reviewer also raised (precision lower bound): "
        f"{sup}/{n_strict}" + (f" ({sup / n_strict:.0%})" if n_strict else ""),
        f"- {precision_line}",
        "",
        "## Category counts",
        "",
        ", ".join(f"{k} {v}" for k, v in Counter(o.category for o in items).most_common()),
    ]
    return "\n".join(lines) + "\n"


def paired(
    base: list[ConcernOutcome], new: list[ConcernOutcome]
) -> tuple[list[ConcernOutcome], list[ConcernOutcome]]:
    """Both runs restricted to the submissions both reviewed (the same concerns)."""
    common = {o.submission_id for o in base} & {o.submission_id for o in new}
    return (
        [o for o in base if o.submission_id in common],
        [o for o in new if o.submission_id in common],
    )


def _cell(items: list[ConcernOutcome]) -> str:
    rc = recall_of(items, "strict")
    return f"{rc.full:.0%} / {rc.with_partial:.0%}" if rc.n else "—"


def _delta(base: list[ConcernOutcome], new: list[ConcernOutcome]) -> str:
    b, n = recall_of(base, "strict"), recall_of(new, "strict")
    if not b.n:
        return "—"
    return f"{(n.full - b.full) * 100:+.0f} / {(n.with_partial - b.with_partial) * 100:+.0f}"


def render_compare(
    name: str,
    base: list[ConcernOutcome],
    new: list[ConcernOutcome],
    base_support: tuple[int, int],
    new_support: tuple[int, int],
) -> list[str]:
    """One comparison block: strict recall (full / incl. partial) per category, overall and by
    scope, on the paired submissions; plus the precision lower bound of each run."""
    b, n = paired(base, new)
    bv = b[0].checker if b else "?"
    nv = n[0].checker if n else "?"
    subs = len({o.submission_id for o in b})
    lines = [
        f"## {name}",
        "",
        f"{subs} submissions reviewed by both, {len(b)} concerns. Strict recall, full match / incl."
        f" partial; Δ in points.",
        "",
        f"| subset | n | {bv} | {nv} | Δ |",
        "|---|---:|---|---|---|",
        f"| **all** | {len(b)} | {_cell(b)} | {_cell(n)} | {_delta(b, n)} |",
    ]
    for scope in ("methodology", "design"):
        bs = [o for o in b if o.scope == scope]
        keys = {x.concern_key for x in bs}
        ns = [o for o in n if o.concern_key in keys]
        lines.append(
            f"| scope {scope} (base run's call) | {len(bs)} | {_cell(bs)} | {_cell(ns)} "
            f"| {_delta(bs, ns)} |"
        )
    for cat in CATEGORIES:
        bc = [o for o in b if o.category == cat]
        nc = [o for o in n if o.category == cat]
        if bc:
            lines.append(f"| {cat} | {len(bc)} | {_cell(bc)} | {_cell(nc)} | {_delta(bc, nc)} |")
    for label, (sup, tot) in ((bv, base_support), (nv, new_support)):
        rate = f" ({sup / tot:.0%})" if tot else ""
        lines.append(
            f"\n- {label}: {tot} strict findings, {sup} also raised by a reviewer{rate} "
            "(precision lower bound)"
        )
    return [*lines, ""]
