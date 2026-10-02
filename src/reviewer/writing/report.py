"""Markdown for one WritingReport and the cross-paper summary with a precision sample."""

import random
from collections import Counter

from reviewer.models import WritingCheck, WritingFinding, WritingReport

CHECK_TITLES = {
    WritingCheck.abstract_overclaim: "Abstract claims more than the results show",
    WritingCheck.unclear_contribution: "Contribution not stated clearly in the introduction",
    WritingCheck.undefined_term: "Terms used before they are defined",
    WritingCheck.caption_not_standalone: "Captions not readable on their own",
    WritingCheck.related_work_list: "Related work lists prior work without engaging",
}


def _one_line(text: str, limit: int = 400) -> str:
    text = " ".join(text.split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_finding(f: WritingFinding, prefix: str = "-") -> list[str]:
    lines = [
        f"{prefix} **{f.location}**: {_one_line(f.detail)}",
        f'  - quote: "{_one_line(f.quote)}"',
    ]
    lines += [f'  - evidence: "{_one_line(e)}"' for e in f.evidence]
    return lines


def render_report(r: WritingReport) -> str:
    u = r.token_usage
    lines = [
        f"# Writing check: {r.title}",
        "",
        f"`{r.paper_id}` · {r.model} · prompt {r.prompt_version} · generated "
        f"{r.generated:%Y-%m-%d %H:%M} UTC · ${u.cost_usd:.2f}"
        + (" (cached)" if u.from_cache else ""),
        "",
        "Every quote below was found verbatim in the parsed paper text by code. Findings whose "
        f"quotes did not verify were dropped ({len(r.dropped)} this paper, listed at the end).",
        "",
        "## Abstract claims traced to results",
        "",
        "| # | abstract claim | verdict | closest result | gap |",
        "|---|---|---|---|---|",
    ]
    for i, t in enumerate(r.abstract_trace, 1):
        lines.append(
            f"| {i} | {_one_line(t.abstract_span, 200)} | {t.verdict.value} | "
            f"{_one_line(t.result_span, 200) or '-'} | {_one_line(t.gap, 200) or '-'} |"
        )
    c = r.contribution
    lines += [
        "",
        "## Contribution statement",
        "",
        f'{c.verdict.value}: "{_one_line(c.intro_span)}"' if c else "No verified intro sentence.",
        "",
        f"Related work: {r.related_work.value if r.related_work else 'not assessed'}.",
        "",
    ]
    for check in WritingCheck:
        found = [f for f in r.findings if f.check == check]
        lines += [f"## {CHECK_TITLES[check]} ({len(found)})", ""]
        if not found:
            lines += ["None.", ""]
            continue
        for f in found:
            lines += render_finding(f)
        lines.append("")
    if r.dropped:
        lines += ["## Dropped (quote did not verify)", ""]
        lines += [f"- {d.check.value}: {d.reason}" for d in r.dropped]
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def stratified_sample(
    reports: list[WritingReport], n: int, seed: int
) -> list[tuple[str, WritingFinding]]:
    """`n` findings spread over the checks: round-robin across checks in a seeded order, each
    check's findings shuffled with the same seed, so rare checks are graded too."""
    rng = random.Random(seed)
    by_check: dict[WritingCheck, list[tuple[str, WritingFinding]]] = {}
    for r in sorted(reports, key=lambda r: r.paper_id):
        for f in r.findings:
            by_check.setdefault(f.check, []).append((r.paper_id, f))
    for items in by_check.values():
        rng.shuffle(items)
    order = [c for c in WritingCheck if c in by_check]
    picked: list[tuple[str, WritingFinding]] = []
    while len(picked) < n and any(by_check[c] for c in order):
        for c in order:
            if by_check[c] and len(picked) < n:
                picked.append(by_check[c].pop())
    return picked


def summary_markdown(reports: list[WritingReport], seed: int, n_sample: int) -> str:
    checks = list(WritingCheck)
    head = "| paper | " + " | ".join(c.value for c in checks) + " | kept | dropped | $ |"
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    totals: Counter[str] = Counter()
    cost = 0.0
    verdicts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    for r in reports:
        by = Counter(f.check.value for f in r.findings)
        totals.update(by)
        totals["kept"] += len(r.findings)
        totals["dropped"] += len(r.dropped)
        cost += r.token_usage.cost_usd
        verdicts.update(t.verdict.value for t in r.abstract_trace)
        reasons.update(f"{d.check.value}: {d.reason}" for d in r.dropped)
        lines.append(
            f"| {r.paper_id} | "
            + " | ".join(str(by.get(c.value, 0)) for c in checks)
            + f" | {len(r.findings)} | {len(r.dropped)} | {r.token_usage.cost_usd:.2f} |"
        )
    lines.append(
        "| **total** | "
        + " | ".join(f"**{totals[c.value]}**" for c in checks)
        + f" | **{totals['kept']}** | **{totals['dropped']}** | **{cost:.2f}** |"
    )
    lines += [
        "",
        "Abstract claims traced: "
        + ", ".join(f"{k} {v}" for k, v in verdicts.most_common())
        + ". Dropped by code: "
        + "; ".join(f"{k} ({v})" for k, v in reasons.most_common())
        + ".",
        "",
    ]
    sample = stratified_sample(reports, n_sample, seed)
    lines += [
        f"#### Precision sample: {len(sample)} findings (seed {seed}, spread across checks)",
        "",
        "Grade each: correct (a real writing problem as described) / wrong.",
        "",
    ]
    for i, (pid, f) in enumerate(sample, 1):
        lines += render_finding(f, prefix=f"{i}. `{pid}` {f.check.value} —")
        lines += ["  - grade: ", ""]
    return "\n".join(lines).rstrip() + "\n"
