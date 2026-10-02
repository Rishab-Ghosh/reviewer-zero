"""Corpus summary of citation reports: per-paper counts and a seeded random sample of flagged
items with evidence, for spot-checking precision. Public and private reports are summarized
separately and never mixed."""

import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from reviewer.citations.checks import not_checked_reason
from reviewer.citations.report import CONTRADICTED_LABEL, FLAG_TITLES, SEVERITY_ORDER
from reviewer.models import CitationReport, SupportVerdict

SAMPLE_KINDS = [SupportVerdict.contradicted.value] + [t.value for t in SEVERITY_ORDER]


@dataclass(frozen=True)
class Item:
    paper_id: str
    kind: str
    lines: list[str]


def load_reports(directory: Path) -> list[CitationReport]:
    return [
        CitationReport.model_validate_json(p.read_text(encoding="utf-8"))
        for p in sorted(directory.glob("*.json"))
    ]


def _flag_items(report: CitationReport) -> list[Item]:
    items = []
    for f in report.flags:
        lines = [f"**{report.paper_id} {f.ref_id} — {FLAG_TITLES[f.type]}:** {f.detail}"]
        lines += [f"  - {e}" for e in f.evidence]
        items.append(Item(report.paper_id, f.type.value, lines))
    return items


def _support_items(report: CitationReport, verdict: SupportVerdict) -> list[Item]:
    refs = {r.ref_id: r for r in report.references}
    items = []
    for s in report.support:
        if s.verdict != verdict:
            continue
        ref = refs.get(s.ref_id)
        r = ref.resolved if ref else None
        lines = [
            f"**{report.paper_id} {s.ref_id} — "
            f"{CONTRADICTED_LABEL if verdict == SupportVerdict.contradicted else verdict.value}**"
            f" ({s.section_heading})",
            f"  - sentence: {s.sentence}",
            f"  - cites: {ref.raw_string if ref else s.ref_id}",
        ]
        if r:
            ids = r.arxiv_id or r.doi or ", ".join(r.openalex_ids)
            lines.append(
                f'  - resolved: "{r.title}" ({r.source}; {ids}; abstract: {r.abstract_source})'
            )
        lines.append(f"  - attributed: {s.attributed}")
        if s.quote:
            lines.append(f'  - abstract says: "{s.quote}"')
        if s.verifier_verdict:
            lines.append(f"  - independent re-judgment: {s.verifier_verdict.value}")
        lines.append(f"  - rationale: {s.rationale}")
        items.append(Item(report.paper_id, verdict.value, lines))
    return items


OTHER_VERDICTS = [
    SupportVerdict.not_in_abstract,
    SupportVerdict.cant_tell_from_abstract,
    SupportVerdict.supported,
]


def count_table(reports: list[CitationReport]) -> list[str]:
    """Columns in severity order: possible contradictions, then reference flags, then the
    arXiv style suggestion, then the other support verdicts."""
    columns = [
        "refs",
        "resolved",
        "not checked",
        CONTRADICTED_LABEL,
        *(t.value for t in SEVERITY_ORDER[:-1]),
        "arxiv_now_published (style)",
        *(v.value for v in OTHER_VERDICTS),
    ]
    head = "| paper | " + " | ".join(columns) + " | LLM $ |"
    lines = [head, "|" + "---|" * (len(columns) + 2)]
    totals: Counter[str] = Counter()
    cost = 0.0
    for rep in reports:
        flags = Counter(f.type for f in rep.flags)
        verdicts = Counter(s.verdict for s in rep.support)
        values = [
            len(rep.references),
            sum(1 for r in rep.references if r.resolved),
            sum(1 for r in rep.references if not_checked_reason(r)),
            verdicts.get(SupportVerdict.contradicted, 0),
            *(flags.get(t, 0) for t in SEVERITY_ORDER),
            *(verdicts.get(v, 0) for v in OTHER_VERDICTS),
        ]
        totals.update(dict(zip(columns, values, strict=True)))
        cost += rep.token_usage.cost_usd
        cells = " | ".join(str(v) for v in values)
        lines.append(f"| {rep.paper_id} | {cells} | {rep.token_usage.cost_usd:.2f} |")
    lines.append(
        "| **total** | " + " | ".join(f"**{totals[c]}**" for c in columns) + f" | **{cost:.2f}** |"
    )
    return lines


def render_summary(reports: list[CitationReport], seed: int, n_sample: int, n_not_in: int) -> str:
    flagged = [i for rep in reports for i in _flag_items(rep)]
    flagged += [i for rep in reports for i in _support_items(rep, SupportVerdict.contradicted)]
    not_in = [i for rep in reports for i in _support_items(rep, SupportVerdict.not_in_abstract)]
    rng = random.Random(seed)
    sample = rng.sample(flagged, min(n_sample, len(flagged)))
    not_in_sample = rng.sample(not_in, min(n_not_in, len(not_in)))
    by_kind = Counter(i.kind for i in flagged)
    lines = [
        f"### Counts ({len(reports)} papers)",
        "",
        *count_table(reports),
        "",
        f"### {len(sample)} random flagged items (seed {seed})",
        "",
        f"Drawn uniformly from {len(flagged)} items: the five reference flags plus contradicted "
        f"({', '.join(f'{k} {by_kind.get(k, 0)}' for k in SAMPLE_KINDS)}).",
        "",
    ]
    for n, item in enumerate(sample, 1):
        lines += [f"{n}. " + item.lines[0], *(f"   {line}" for line in item.lines[1:]), ""]
    lines += [
        f"### {len(not_in_sample)} random not_in_abstract items (seed {seed}), shown separately",
        "",
        f"Not accusations: the abstract does not address the attribution. {len(not_in)} in total.",
        "",
    ]
    for n, item in enumerate(not_in_sample, 1):
        lines += [f"{n}. " + item.lines[0], *(f"   {line}" for line in item.lines[1:]), ""]
    return "\n".join(lines).rstrip() + "\n"
