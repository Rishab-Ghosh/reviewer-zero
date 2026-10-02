"""Markdown rendering of a CitationReport. Leads with contradicted citations."""

from collections import Counter

from reviewer.citations.checks import not_checked_reason
from reviewer.models import (
    CitationFlag,
    CitationReport,
    CitationSupport,
    FlagType,
    ReferenceCheck,
    SupportVerdict,
)

FLAG_TITLES = {
    FlagType.duplicate: "Duplicate references",
    FlagType.wrong_authors: "Wrong authors",
    FlagType.wrong_year: "Wrong year",
    FlagType.unresolved: "Unresolved references",
    FlagType.arxiv_now_published: "Style suggestion: cited as arXiv, since published at a venue",
}
SEVERITY_ORDER = [
    FlagType.duplicate,
    FlagType.wrong_authors,
    FlagType.wrong_year,
    FlagType.unresolved,
    FlagType.arxiv_now_published,
]
"""Reference flags, most severe first; contradicted citations come before all of them."""
CONTRADICTED_LABEL = "possible contradiction, check"
"""How contradicted verdicts are shown until their precision is confirmed."""


def verdict_label(v: SupportVerdict) -> str:
    return CONTRADICTED_LABEL if v == SupportVerdict.contradicted else v.value


def _cell(text: str, limit: int = 220) -> str:
    text = " ".join(text.split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _quote_block(text: str) -> str:
    return "> " + " ".join(text.split())


def verdict_counts(report: CitationReport) -> Counter[SupportVerdict]:
    return Counter(s.verdict for s in report.support)


def flag_counts(report: CitationReport) -> Counter[FlagType]:
    return Counter(f.type for f in report.flags)


def _summary(report: CitationReport) -> list[str]:
    refs = report.references
    by_source = Counter(r.resolved.source for r in refs if r.resolved)
    flags = flag_counts(report)
    verdicts = verdict_counts(report)
    with_abstract = sum(1 for r in refs if r.resolved and r.resolved.abstract)
    overridden = Counter(
        s.override_reason.split(" (")[0] for s in report.support if s.override_reason
    )
    lines = [
        "## Summary",
        "",
        "| | count |",
        "|---|---:|",
        f"| references | {len(refs)} |",
        f"| resolved | {sum(by_source.values())} "
        f"({', '.join(f'{k} {v}' for k, v in sorted(by_source.items())) or 'none'}) |",
        f"| resolved with an abstract | {with_abstract} |",
    ]
    not_checked = [r for r in refs if not_checked_reason(r)]
    lines.append(
        f"| support: {CONTRADICTED_LABEL} | {verdicts.get(SupportVerdict.contradicted, 0)} |"
    )
    lines += [f"| flag: {t.value} | {flags.get(t, 0)} |" for t in SEVERITY_ORDER]
    lines.append(f"| unresolved, not checked (web page / skipped lookup) | {len(not_checked)} |")
    lines += [
        f"| support: {v.value} | {verdicts.get(v, 0)} |"
        for v in SupportVerdict
        if v != SupportVerdict.contradicted
    ]
    lines += [
        *(f"| verdict overridden: {why} | {n} |" for why, n in sorted(overridden.items())),
        f"| errors | {len(report.errors)} |",
        "",
    ]
    return lines


def _support_item(s: CitationSupport, ref: ReferenceCheck | None) -> list[str]:
    lines = [
        f"### {s.ref_id} — {_cell(s.section_heading, 80)}",
        "",
        _quote_block(s.sentence),
        "",
        f"- **Cites:** {ref.raw_string if ref else s.ref_id}",
    ]
    if ref and ref.resolved:
        ids = ref.resolved.arxiv_id or ref.resolved.doi or ", ".join(ref.resolved.openalex_ids)
        lines.append(
            f'- **Resolved:** "{ref.resolved.title}" ({ref.resolved.source}; {ids}; '
            f"abstract from {ref.resolved.abstract_source})"
        )
    if s.verifier_verdict:
        lines.append(f"- **Independent re-judgment:** {s.verifier_verdict.value}")
    lines.append(f"- **Attributed:** {s.attributed}")
    if s.property_type:
        lines.append(f"- **Property type:** {s.property_type.value}")
    if s.sentence_span:
        lines.append(f'- **Sentence span:** "{s.sentence_span}"')
    lines += [
        f'- **Abstract span:** "{s.quote}"',
        f"- **Explanation:** {s.rationale}",
        "",
    ]
    return lines


def _flag_item(flag: CitationFlag) -> list[str]:
    lines = [f"- **{flag.ref_id}**: {flag.detail}"]
    lines += [f"  - {_cell(e, 400)}" for e in flag.evidence]
    return lines


def _support_table(items: list[CitationSupport], refs: dict[str, ReferenceCheck]) -> list[str]:
    lines = ["| ref | sentence | attributed | rationale |", "|---|---|---|---|"]
    for s in items:
        ref = refs.get(s.ref_id)
        label = s.ref_id + (f" ({_cell(ref.resolved.title, 60)})" if ref and ref.resolved else "")
        note = s.rationale
        if s.overridden_from:
            note = f"[judge said {s.overridden_from.value}; {s.override_reason}] {note}"
        lines.append(
            f"| {label} | {_cell(s.sentence)} | {_cell(s.attributed, 120)} | {_cell(note)} |"
        )
    return [*lines, ""]


def render_report(report: CitationReport) -> str:
    refs = {r.ref_id: r for r in report.references}
    index_state = "up" if report.index_available else "not running (skipped)"
    lines = [
        f"# Citation check: {report.title}",
        "",
        f"`{report.paper_id}` · generated {report.generated:%Y-%m-%d %H:%M} UTC · "
        f"resolution: reviewer-index ({index_state}) → OpenAlex → Crossref · "
        f"support judge: {report.model} (`{report.prompt_version}`), abstract only · "
        f"LLM cost ${report.token_usage.cost_usd:.2f}",
        "",
        "Every check compares against the reference as printed (GROBID's consolidated fields are "
        "only lookup hints). `not_in_abstract` means the abstract does not address the claim, "
        "not that the paper does not support it.",
        "",
        *_summary(report),
    ]

    contradicted = [s for s in report.support if s.verdict == SupportVerdict.contradicted]
    lines += [
        f"## Possible contradictions, check ({len(contradicted)})",
        "",
        "The cited abstract may state a property incompatible with what the sentence attributes "
        "(method type, who does what, result direction, or a number). Passed code checks and an "
        "independent re-judgment; precision is not yet established, so verify each by hand.",
        "",
    ]
    if not contradicted:
        lines += ["None.", ""]
    for s in contradicted:
        lines += _support_item(s, refs.get(s.ref_id))

    dropped = [
        s
        for s in report.support
        if s.overridden_from == SupportVerdict.contradicted
        and s.verdict == SupportVerdict.not_in_abstract
    ]
    lines += [
        f"### Called contradicted by the judge, then dropped ({len(dropped)})",
        "",
        "Failed a code check or the independent re-judgment; counted as not_in_abstract.",
        "",
    ]
    for s in dropped:
        lines += _support_item(s, refs.get(s.ref_id))
        lines.insert(-1, f"- **Dropped because:** {s.override_reason}")
    if not dropped:
        lines += ["None.", ""]

    for t in SEVERITY_ORDER:
        flags = [f for f in report.flags if f.type == t]
        lines += [f"## {FLAG_TITLES[t]} ({len(flags)})", ""]
        if t == FlagType.arxiv_now_published:
            lines += ["Not an error: the reference is correct but a venue version exists.", ""]
        for f in flags:
            lines += _flag_item(f)
        lines += [""] if flags else ["None.", ""]
        if t == FlagType.unresolved:
            not_checked = [(r, why) for r in report.references if (why := not_checked_reason(r))]
            lines += [f"### Unresolved but not checked ({len(not_checked)})", ""]
            lines += [
                f"- **{r.ref_id}** ({why}): {_cell(r.raw_string, 300)}" for r, why in not_checked
            ]
            lines.append("")

    not_in = [s for s in report.support if s.verdict == SupportVerdict.not_in_abstract]
    lines += [f"## Not addressed by the abstract ({len(not_in)})", ""]
    lines += _support_table(not_in, refs) if not_in else ["None.", ""]

    cant = [s for s in report.support if s.verdict == SupportVerdict.cant_tell_from_abstract]
    no_abs = [s for s in cant if not s.judged_by_llm]
    judged = [s for s in cant if s.judged_by_llm]
    lines += [
        f"## Can't tell from abstract ({len(cant)})",
        "",
        f"- {len(no_abs)} citation(s) whose reference has no abstract from any source: "
        + (", ".join(sorted({s.ref_id for s in no_abs})) or "none"),
        f"- {len(judged)} judged by the LLM (abstract unusable, or verdict overridden):",
        "",
    ]
    if judged:
        lines += _support_table(judged, refs)

    supported = [s for s in report.support if s.verdict == SupportVerdict.supported]
    lines += [f"## Supported ({len(supported)})", ""]
    lines += _support_table(supported, refs) if supported else ["None.", ""]

    if report.errors:
        lines += ["## Errors", "", *(f"- {_cell(e, 300)}" for e in report.errors), ""]
    return "\n".join(lines).rstrip() + "\n"
