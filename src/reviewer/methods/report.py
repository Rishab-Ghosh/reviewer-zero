"""Markdown for methodology reviews, the critique evaluation, and the precision sample. Pure."""

from collections import Counter

from reviewer.methods.evaluate import CritiqueOutcome, Raised, Recall, recall
from reviewer.models import (
    CheckedConcern,
    CheckedFinding,
    ChecklistItemId,
    ConcernCheck,
    ConcernValidity,
    CritiqueScope,
    MethodsVerdict,
    PaperMethodsReview,
)

TIER_ORDER = ("own", "lab", "ordinary", "famous")
VERDICTS = list(MethodsVerdict)


def _quote_lines(quotes: list[tuple[str, bool]]) -> list[str]:
    return [f"  > {q}" + ("" if ok else "  *(not found in paper)*") for q, ok in quotes]


def verdict_table(reviews: list[PaperMethodsReview]) -> list[str]:
    lines = [
        "| item | " + " | ".join(v.value for v in VERDICTS) + " | downgraded |",
        "|---|" + "---:|" * (len(VERDICTS) + 1),
    ]
    present = {f.item for r in reviews for f in r.findings}
    for item in [i for i in ChecklistItemId if i in present]:
        fs = [f for r in reviews for f in r.findings if f.item == item]
        counts = Counter(f.verdict for f in fs)
        lines.append(
            f"| {item.value} | "
            + " | ".join(str(counts.get(v, 0)) for v in VERDICTS)
            + f" | {sum(f.downgraded for f in fs)} |"
        )
    return lines


def quote_stats(reviews: list[PaperMethodsReview]) -> dict[str, int]:
    qs = [q for r in reviews for f in r.findings for q in f.quotes]
    qs += [q for r in reviews for c in r.other_concerns for q in c.quotes]
    return {
        "quotes": len(qs),
        "verified": sum(q.verified for q in qs),
        "repaired_findings": sum(f.repaired for r in reviews for f in r.findings)
        + sum(c.repaired for r in reviews for c in r.other_concerns),
        "downgraded": sum(f.downgraded for r in reviews for f in r.findings),
        "other_concerns": sum(len(r.other_concerns) for r in reviews),
        "other_unverified": sum(not c.verified for r in reviews for c in r.other_concerns),
    }


def _finding_lines(f: CheckedFinding) -> list[str]:
    tag = " — downgraded, no verified quote" if f.downgraded else ""
    return [
        f"- **{f.item.value}: {f.verdict.value}**{tag}. {f.rationale}",
        *_quote_lines([(q.quote, q.verified) for q in f.quotes]),
    ]


def _concern_lines(i: int, c: CheckedConcern) -> list[str]:
    tag = "" if c.verified else " *(unverified: counts as unclear)*"
    return [f"{i}. {c.concern}{tag}", *_quote_lines([(q.quote, q.verified) for q in c.quotes])]


def render_review(r: PaperMethodsReview, title: str) -> list[str]:
    lines = [f"## {r.paper_id} · {r.tier} · {title}", ""]
    lines += ["| item | verdict | concern |", "|---|---|---|"]
    for f in r.findings:
        v = f.verdict.value + (f" (model: {f.model_verdict.value})" if f.downgraded else "")
        if f.agreement:
            v += f" · {f.agreement}"
        lines.append(f"| {f.item.value} | {v} | {f.concern.replace('|', '/') or '—'} |")
    lines.append("")
    if r.runs == 2:
        for label, head in (
            ("confirmed", "### Confirmed findings (raised in both runs)"),
            ("possible", "### Possible findings (raised in only one of two runs)"),
        ):
            fs = [f for f in r.findings if f.agreement == label]
            cs = [c for c in r.other_concerns if c.agreement == label]
            lines += [head, ""]
            if not fs and not cs:
                lines += ["None.", ""]
                continue
            for f in fs:
                lines += _finding_lines(f)
            for i, c in enumerate(cs, start=1):
                lines += _concern_lines(i, c)
            lines.append("")
        rest = [f for f in r.findings if not f.agreement]
        if rest:
            lines += ["### Other checklist items", ""]
            for f in rest:
                lines += _finding_lines(f)
        return [*lines, ""]
    for f in r.findings:
        lines += _finding_lines(f)
    if r.other_concerns:
        lines += ["", "**Other concerns:**", ""]
        for i, c in enumerate(r.other_concerns, start=1):
            lines += _concern_lines(i, c)
    return [*lines, ""]


def render_reviews(reviews: list[PaperMethodsReview], titles: dict[str, str]) -> str:
    order = sorted(
        reviews,
        key=lambda r: (TIER_ORDER.index(r.tier) if r.tier in TIER_ORDER else 9, r.paper_id),
    )
    stats = quote_stats(reviews)
    cost = sum(r.token_usage.cost_usd for r in reviews)
    lines = [
        "# Methodology checker report (private)",
        "",
        f"{len(reviews)} papers; {reviews[0].model}, {reviews[0].prompt_version}, effort "
        f"{reviews[0].effort}. Recorded cost ${cost:.2f} (first-run cost, cached reruns "
        "included at original cost).",
        "",
        "For every item, yes = good practice; no = a reviewer should object; na = does not "
        "apply. A verdict with no quote found in the paper is downgraded to unclear.",
        "",
        f"Quotes: {stats['verified']}/{stats['quotes']} verified. Findings sent to repair: "
        f"{stats['repaired_findings']}. Downgraded: {stats['downgraded']}. Other concerns: "
        f"{stats['other_concerns']} ({stats['other_unverified']} unverified).",
        "",
        *verdict_table(reviews),
        "",
    ]
    for r in order:
        lines += render_review(r, titles.get(r.paper_id, ""))
    return "\n".join(lines) + "\n"


def _recall_row(label: str, rc: Recall) -> str:
    return (
        f"| {label} | {rc.n} | {rc.match} | {rc.partial} | {rc.full:.0%} | {rc.with_partial:.0%} |"
    )


def render_eval(
    items: list[CritiqueOutcome],
    checks: list[tuple[Raised, ConcernCheck]],
    missing_reviews: list[str],
    cost: float,
) -> str:
    lines = [
        "# Methodology checker vs human critiques (private)",
        "",
        f"{len(items)} critiques across {len({o.paper_id for o in items})} papers. Matcher and "
        f"verifier spend this run: ${cost:.2f}.",
        "",
        "- **strict:** only verified `no` findings and verified other concerns count as raised.",
        "- **lenient:** also `unclear` findings (including downgraded ones) and unverified other "
        "concerns.",
        "- **scope:** the matcher's call on whether the critique is about experimental "
        "methodology or about the method's design/applicability (which a methodology checklist "
        "is not meant to raise).",
        "",
        "## Recall",
        "",
        "| subset | n | match | partial | recall | recall incl. partial |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in ("strict", "lenient"):
        lines.append(_recall_row(f"all, {mode}", recall(items, mode)))
        for scope in CritiqueScope:
            lines.append(_recall_row(f"{scope.value}, {mode}", recall(items, mode, scope)))
    if missing_reviews:
        lines += ["", f"Papers with critiques but no review: {', '.join(missing_reviews)}."]
    lines += ["", "## Per critique", ""]
    lines += ["| paper | # | scope | strict | lenient | findings | critique | matcher |"]
    lines += ["|---|---:|---|---|---|---|---|---|"]
    for o in items:
        crit = o.text.replace("|", "/")
        lines.append(
            f"| {o.paper_id} | {o.index} | {o.scope.value} | {o.strict.value} | "
            f"{o.lenient.value} | {', '.join(o.finding_ids) or '—'} | {crit} | "
            f"{o.reason.replace('|', '/')} |"
        )
    lines += ["", "## Precision sample (LLM verifier with the full paper)", ""]
    if not checks:
        lines.append("Not run.")
    else:
        counts = Counter(c.validity for _, c in checks)
        n = len(checks)
        valid = counts.get(ConcernValidity.valid, 0)
        partly = counts.get(ConcernValidity.partly_valid, 0)
        lines.append(
            f"{n} strict concerns sampled (seeded) across all papers: "
            + ", ".join(f"{v.value} {counts.get(v, 0)}" for v in ConcernValidity)
            + f". Precision (valid) {valid / n:.0%}; valid + partly valid "
            f"{(valid + partly) / n:.0%}."
        )
        lines += ["", "| concern | verifier | checker's concern | verifier's reason |"]
        lines += ["|---|---|---|---|"]
        for r, c in checks:
            lines.append(
                f"| {r.id} | {c.validity.value} | {r.text.replace('|', '/')} | "
                f"{c.reason.replace('|', '/')} |"
            )
    return "\n".join(lines) + "\n"


def render_precision_grading(sample: list[Raised]) -> str:
    lines = [
        "# Methodology concerns: precision sample for human grading",
        "",
        "For each concern write valid / partly_valid / invalid after the colon: does the paper",
        "really have this problem as stated? The verifier's labels are not in this file.",
    ]
    for k, r in enumerate(sample, start=1):
        lines += ["", f"## M{k} · {r.id}", "", r.text, ""]
        lines += [f"> {q}" for q in r.quotes] or ["> (no verified quote)"]
        lines += ["", "- validity: ", "- notes: "]
    return "\n".join(lines) + "\n"
