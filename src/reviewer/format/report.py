"""Markdown rendering of a FormatReport: violations first, each with its rule's source."""

from reviewer.format.venues import VENUES
from reviewer.models import FormatReport, FormatSeverity

SEVERITY_ORDER = [FormatSeverity.violation, FormatSeverity.warning, FormatSeverity.info]


def _cell(text: str, limit: int = 160) -> str:
    text = " ".join(text.split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_report(r: FormatReport) -> str:
    d = r.declared
    venue = f"{d.venue.value} {d.year or ''}".strip() if d.venue else "none recognized"
    lines = [
        f"# Format check: {r.title}",
        "",
        f"`{r.paper_id}` · {r.n_pages} PDF pages · generated {r.generated:%Y-%m-%d %H:%M} UTC · "
        "no LLM",
        "",
        f"- **Declared venue:** {venue}; **stage:** {d.stage.value}"
        + (f' (page 1: "{_cell(d.text, 120)}")' if d.text else ""),
        "- **Rules:** "
        + ", ".join(v.edition for v in VENUES.values())
        + ", checked as current rules even for papers from older editions. Violations are "
        "raised only for the declared venue (and, for anonymization, only at the submission "
        "stage); everything else is info.",
        "",
        "## Main-text pages by venue",
        "",
        "| venue | main text | pages used | submission limit | camera-ready limit | ends at |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for p in r.pages:
        lines.append(
            f"| {p.venue.value} | {p.main_text_pages:.2f} | {p.pages_used} | "
            f"{p.submission_limit} | {p.camera_ready_limit} | {_cell(p.ends_at or 'not found')} |"
        )
    lines.append("")
    for sev in SEVERITY_ORDER:
        found = [f for f in r.findings if f.severity == sev]
        lines += [f"## {sev.value.capitalize()}s ({len(found)})", ""]
        if not found:
            lines += ["None.", ""]
        for f in found:
            where = f" p{f.page}" if f.page else ""
            venue_tag = f" [{f.venue.value}]" if f.venue else ""
            lines.append(f"- **{f.check.value}**{venue_tag}{where}: {f.detail}")
            lines += [f"  - {_cell(e, 300)}" for e in f.evidence]
            if f.source:
                lines.append(f"  - source: {f.source}")
        lines.append("")
    if r.errors:
        lines += ["## Errors", "", *(f"- {e}" for e in r.errors), ""]
    return "\n".join(lines).rstrip() + "\n"
