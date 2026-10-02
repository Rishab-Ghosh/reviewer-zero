"""Document layout from the PDF text layer: pages, section headings, where the main text ends,
and the venue/stage the PDF declares on page 1. Pure functions over page texts."""

import re
from dataclasses import dataclass

from reviewer.format.venues import VenueRules
from reviewer.models import DeclaredVenue, SubmissionStage, Venue, VenuePages

# kind -> (pattern on the normalized heading text, run-in allowed)
# Run-in: the heading may be followed by text on the same line ("Acknowledgements We thank").
HEADING_KINDS: dict[str, tuple[re.Pattern[str], bool]] = {
    "references": (re.compile(r"^(references|bibliography|literature cited)$"), False),
    "appendix": (
        re.compile(r"^(appendix|appendices|supplementary materials?|technical appendices)\b"),
        False,
    ),
    "acknowledgments": (re.compile(r"^(acknowledge?ments?|funding)\b"), True),
    "ethics": (re.compile(r"^ethics( statement| considerations)?\b"), True),
    "reproducibility": (re.compile(r"^reproducibility statement\b"), True),
    "ai_use": (
        re.compile(
            r"^((ai|llm|large language models?) (use|usage)( statement| disclosure)?"
            r"|(the )?use of (large )?language models( \(llms\))?|llm usage)\b"
        ),
        True,
    ),
    "impact": (
        re.compile(r"^(impact statement|broader impacts?( statement)?|societal impacts?)\b"),
        True,
    ),
    "checklist": (re.compile(r"^((neurips )?paper checklist|checklist)\b"), True),
    "limitations": (re.compile(r"^limitations?\b"), True),
}
NUMBERING_RE = re.compile(r"^(\d+(\.\d+)*|[A-Z](\.\d+)*)\.?\s+(?=\S)")
SMALL_CAPS_RE = re.compile(r"\b([A-Z]) (?=[A-Z]{2,})")
PROSE_RE = re.compile(r"\b(contains?|shows?|provides?|see|details|describes?|we|is|are)\b")


@dataclass(frozen=True)
class Heading:
    kind: str
    page: int
    """1-based page number."""
    line: int
    """Index into the page's non-empty lines."""
    fraction: float
    """Share of the page's non-empty lines above the heading."""
    text: str


def page_lines(page: str) -> list[str]:
    return [ln.strip() for ln in page.splitlines() if ln.strip()]


def heading_candidates(line: str) -> list[str]:
    """Normalized forms of a line to test as a heading: as is, with small-caps splits joined
    ('R EFERENCES' -> 'REFERENCES'), each without a leading section number."""
    forms = []
    for variant in (line, SMALL_CAPS_RE.sub(r"\1", line)):
        for v in (variant, NUMBERING_RE.sub("", variant)):
            v = " ".join(v.split()).lower().strip(" .:")
            if v and v not in forms:
                forms.append(v)
    return forms


def _run_in_ok(line: str, pattern: re.Pattern[str]) -> bool:
    """A run-in heading must be followed by '.', ':' or a capitalized word ('Acknowledgements
    We thank ...', 'Limitations. Our ...'), not by prose ('Limitations of prior work are')."""
    flat = SMALL_CAPS_RE.sub(r"\1", " ".join(line.split()))
    flat = NUMBERING_RE.sub("", flat)
    m = re.match(pattern.pattern.lstrip("^"), flat, re.IGNORECASE)
    if not m:
        return False
    after = flat[m.end() :]
    stripped = after.lstrip()
    return not stripped or after[:1] in ".:" or stripped[:1] in ".:&" or stripped[:1].isupper()


def _appendix_title_ok(line: str, rest: str) -> bool:
    """'Appendix A', 'Appendix B.2: Proofs', 'Appendix A Additional Results' are headings;
    'Appendix B.1.2 contains further details' is prose."""
    if PROSE_RE.search(rest) or len(rest.split()) > 8 or re.search(r"[),;]", rest):
        return False
    words = [w for w in line.split()[1:] if w[:1].isalpha()]
    label_first = bool(re.match(r"^[a-z0-9]{1,2}(\.\d+)*\b", rest))
    titled = sum(w[:1].isupper() for w in words) >= 0.6 * max(1, len(words))
    return (label_first and len(rest.split()) == 1) or titled


def classify_heading(line: str) -> str | None:
    if len(line) > 160:
        return None
    for form in heading_candidates(line):
        for kind, (pattern, run_in) in HEADING_KINDS.items():
            m = pattern.match(form)
            if not m:
                continue
            rest = form[m.end() :].strip(" .:")
            if kind == "appendix":
                if rest and not _appendix_title_ok(line, rest):
                    continue  # "Appendix B.1.2 contains further details"
                return kind
            if rest and not run_in:
                continue
            if rest and not _run_in_ok(line, pattern):
                continue
            return kind
    return None


DOT_LEADER_RE = re.compile(r"(\.\s?){4,}\s*\d+\s*$")
SENTENCE_END_RE = re.compile(r"[.:!?)\]\"”]$")


def is_toc_page(lines: list[str]) -> bool:
    """A table of contents: several lines with dot leaders and a page number."""
    return sum(1 for ln in lines if DOT_LEADER_RE.search(ln)) >= 5


def find_headings(pages: list[str]) -> list[Heading]:
    out = []
    for p, page in enumerate(pages, 1):
        lines = page_lines(page)
        if is_toc_page(lines):
            continue  # "Appendices ... 20" in a table of contents is not a heading
        for i, line in enumerate(lines):
            kind = classify_heading(line)
            if not kind:
                continue
            # "... details in\nAppendix A." is prose wrapped onto its own line
            if kind == "appendix" and i > 0 and not SENTENCE_END_RE.search(lines[i - 1]):
                continue
            out.append(Heading(kind, p, i, i / max(1, len(lines)), line[:120]))
    return out


def main_text_pages(pages: list[str], headings: list[Heading], rules: VenueRules) -> VenuePages:
    """The main text ends at the first heading (after page 1) whose content the venue excludes
    from the page limit."""
    ends = [h for h in headings if h.page >= 2 and h.kind in rules.excluded_headings]
    if ends:
        end = min(ends, key=lambda h: (h.page, h.line))
        used = end.page - 1 + (1 if end.fraction > 0.02 else 0)
        main = end.page - 1 + end.fraction
        ends_at = f"{end.text} (p{end.page})"
    else:
        used, main, ends_at = len(pages), float(len(pages)), None
    return VenuePages(
        venue=rules.venue,
        submission_limit=rules.submission_pages,
        camera_ready_limit=rules.camera_ready_pages,
        main_text_pages=round(main, 2),
        pages_used=used,
        ends_at=ends_at,
    )


def references_span(pages: list[str], headings: list[Heading]) -> tuple[int, int] | None:
    """(first page, last page) of the reference list: from the references heading to the page
    before the next appendix/checklist heading, or the end."""
    refs = [h for h in headings if h.kind == "references" and h.page >= 2]
    if not refs:
        return None
    start = refs[0]
    after = [
        h
        for h in headings
        if h.kind in ("appendix", "checklist") and (h.page, h.line) > (start.page, start.line)
    ]
    end_page = after[0].page if after else len(pages)
    return start.page, end_page


# --------------------------------------------------------------------------- declared venue

ICLR_RE = re.compile(r"(Published|Under review) as a conference paper at ICLR (\d{4})")
NEURIPS_RE = re.compile(
    r"(Submitted to )?(\d+)(?:st|nd|rd|th) Conference on Neural Information Processing\s+"
    r"Systems \((?:NeurIPS|NIPS) (\d{4})\)"
)
ICML_RE = re.compile(
    r"Proceedings of the (\d+)(?:st|nd|rd|th) International Conference on Machine\s+Learning"
)
ICML_REVIEW_RE = re.compile(r"Under review by the International Conference on Machine Learning")
ANON_RE = re.compile(r"Anonymous Authors?(\(s\))?", re.IGNORECASE)
WORKSHOP_RE = re.compile(r"\bworkshop\b", re.IGNORECASE)


def _line_with(text: str, start: int) -> str:
    a = text.rfind("\n", 0, start) + 1
    b = text.find("\n", start)
    return " ".join(text[a : b if b != -1 else None].split())[:160]


def declared_venue(page1: str) -> DeclaredVenue:
    """Venue and stage printed on page 1 by the style file (running header/footer)."""
    flat = " ".join(page1.split())
    anon = ANON_RE.search(flat) is not None
    if m := ICLR_RE.search(flat):
        stage = (
            SubmissionStage.camera_ready
            if m.group(1) == "Published"
            else (SubmissionStage.submission)
        )
        return DeclaredVenue(venue=Venue.iclr, stage=stage, year=int(m.group(2)), text=m.group(0))
    if m := NEURIPS_RE.search(flat):
        line = _line_with(page1, page1.find(m.group(2)))
        if WORKSHOP_RE.search(line) or WORKSHOP_RE.search(flat[max(0, m.start() - 80) : m.start()]):
            return DeclaredVenue(venue=None, stage=SubmissionStage.preprint, text=m.group(0))
        stage = (
            SubmissionStage.submission if (m.group(1) or anon) else (SubmissionStage.camera_ready)
        )
        return DeclaredVenue(
            venue=Venue.neurips, stage=stage, year=int(m.group(3)), text=m.group(0)
        )
    if m := ICML_RE.search(flat):
        return DeclaredVenue(
            venue=Venue.icml,
            stage=SubmissionStage.camera_ready,
            year=1983 + int(m.group(1)),
            text=m.group(0),
        )
    if m := ICML_REVIEW_RE.search(flat):
        return DeclaredVenue(venue=Venue.icml, stage=SubmissionStage.submission, text=m.group(0))
    if anon:
        return DeclaredVenue(
            venue=None, stage=SubmissionStage.submission, text=ANON_RE.search(flat).group(0)
        )  # type: ignore[union-attr]
    return DeclaredVenue(venue=None, stage=SubmissionStage.preprint)
