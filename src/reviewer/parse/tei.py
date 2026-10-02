"""Pure TEI (GROBID output) → ParsedPaper conversion. No I/O, no network."""

import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from lxml import etree

from reviewer.models import (
    Author,
    Figure,
    InTextCitation,
    ParsedPaper,
    ParseReport,
    Reference,
    Section,
    Table,
)

logger = logging.getLogger(__name__)

TEI_NS = "http://www.tei-c.org/ns/1.0"
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
NS = {"tei": TEI_NS}

_Element = etree._Element

# Leading "A", "B.1", "C.1.2" appendix numbers GROBID leaves inside the heading text.
_DOTTED_LETTER_NUM = re.compile(r"^([A-Z](?:\.\d+)+)\.?\s+(\S.*)$")
_SINGLE_LETTER_NUM = re.compile(r"^([A-Z])\.?\s+([A-Z0-9].*)$")
_ARABIC_NUM = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(\S.*)$")
_CAPTION_LABEL_PREFIX = re.compile(
    r"^(?:Figure|Fig\.?|Table|Tab\.?)\s*[A-Z]?\d+[a-z]?\s*[:.]\s*", re.I
)
_ARXIV_IN_TEXT = re.compile(r"(?:arXiv:\s*|abs/)(\d{4}\.\d{4,5})(?:v\d+)?", re.I)
_ARXIV_ID = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?")
_YEAR = re.compile(r"^(\d{4})")
_UNRESOLVED_WARNING_LIMIT = 20


def _tag(name: str) -> str:
    return f"{{{TEI_NS}}}{name}"


def _is_ref(el: _Element | None) -> bool:
    return el is not None and el.tag == _tag("ref")


def normalize_ws(text: str) -> str:
    return " ".join(text.split())


def element_text(el: _Element, skip: Callable[[_Element], bool] | None = None) -> str:
    """All descendant text of `el`, whitespace-normalised, omitting subtrees where `skip` is true.

    Adjacent <s> sentences are separated by a space (GROBID emits them with no whitespace).
    """
    parts: list[str] = []

    def walk(node: _Element) -> None:
        if node.text:
            parts.append(node.text)
        for child in node:
            if not isinstance(child.tag, str):  # comments, processing instructions
                if child.tail:
                    parts.append(child.tail)
                continue
            if skip is None or not skip(child):
                if child.tag == _tag("s"):
                    parts.append(" ")
                walk(child)
                if child.tag == _tag("s"):
                    parts.append(" ")
            if child.tail:
                parts.append(child.tail)
            elif child.tag == _tag("ref") and _is_ref(child.getnext()):
                # GROBID drops the space in '[35, 2, 5]' when splitting it into adjacent refs
                parts.append(" ")

    walk(el)
    return normalize_ws("".join(parts))


def _is_footnote_marker(el: _Element) -> bool:
    return el.tag == _tag("ref") and el.get("type") == "foot"


def body_text(el: _Element) -> str:
    """Running-text view of an element: footnote markers (e.g. 'foot_0') removed."""
    return element_text(el, skip=_is_footnote_marker)


# --------------------------------------------------------------------------- header


def _first(el: _Element | None, path: str) -> _Element | None:
    if el is None:
        return None
    found = el.find(path, NS)
    return found


def extract_title(root: _Element) -> str:
    title_stmt = root.find(".//tei:teiHeader/tei:fileDesc/tei:titleStmt", NS)
    if title_stmt is None:
        return ""
    title = _first(title_stmt, "tei:title[@type='main']")
    if title is None:
        title = _first(title_stmt, "tei:title")
    return element_text(title) if title is not None else ""


def person_name(pers: _Element) -> str:
    names = [element_text(n) for n in pers.findall("tei:forename", NS)]
    names += [element_text(n) for n in pers.findall("tei:surname", NS)]
    joined = " ".join(n for n in names if n)
    return joined or element_text(pers)


def affiliation_text(aff: _Element) -> str | None:
    orgs: list[str] = []
    for org in aff.findall("tei:orgName", NS):
        text = element_text(org)
        if text and text not in orgs:
            orgs.append(text)
    country = _first(aff, "tei:address/tei:country")
    if country is not None and (c := element_text(country)) and c not in orgs:
        orgs.append(c)
    return ", ".join(orgs) or None


def extract_authors(root: _Element) -> list[Author]:
    analytic = root.find(
        ".//tei:teiHeader/tei:fileDesc/tei:sourceDesc/tei:biblStruct/tei:analytic", NS
    )
    if analytic is None:
        return []
    authors: list[Author] = []
    for author in analytic.findall("tei:author", NS):
        pers = author.find("tei:persName", NS)
        if pers is None:
            continue
        name = person_name(pers)
        if not name:
            continue
        aff = author.find("tei:affiliation", NS)
        authors.append(
            Author(name=name, affiliation=affiliation_text(aff) if aff is not None else None)
        )
    return authors


def extract_abstract(root: _Element) -> str:
    abstract = root.find(".//tei:teiHeader/tei:profileDesc/tei:abstract", NS)
    if abstract is None:
        return ""
    paragraphs = [body_text(p) for p in abstract.iter(_tag("p"))]
    paragraphs = [p for p in paragraphs if p]
    if not paragraphs:
        return body_text(abstract)
    return "\n\n".join(paragraphs)


# --------------------------------------------------------------------------- sections


def level_from_number(number: str | None) -> int:
    if not number:
        return 1
    return number.count(".") + 1


def split_heading_number(
    heading: str, n_attr: str | None, in_back: bool
) -> list[tuple[str, str | None]]:
    """Return one or two (heading, number) pairs for a TEI <head>.

    Handles GROBID's quirks: a number left inside the heading text ('B.1 Agent: ...'),
    and a section heading merged with its first subsection ('Data 4.1 Agent Inputs', n='4').
    """
    heading = normalize_ws(heading).rstrip(":").strip()
    number = n_attr.strip().rstrip(".") if n_attr else None
    if number:
        merged = re.match(rf"^(.+?)\s+({re.escape(number)}\.\d+)\s+(\S.*)$", heading)
        if merged:
            return [(merged.group(1), number), (merged.group(3), merged.group(2))]
        return [(heading, number)]
    for pattern in (_ARABIC_NUM, _DOTTED_LETTER_NUM):
        m = pattern.match(heading)
        if m:
            return [(m.group(2), m.group(1))]
    if in_back:
        m = _SINGLE_LETTER_NUM.match(heading)
        if m:
            return [(m.group(2), m.group(1))]
    return [(heading, None)]


@dataclass
class _SectionBuild:
    sections: list[Section] = field(default_factory=list)
    citations: list[InTextCitation] = field(default_factory=list)


def _bibr_targets(el: _Element) -> list[str]:
    ids: list[str] = []
    for ref in el.iter(_tag("ref")):
        if ref.get("type") != "bibr":
            continue
        for target in (ref.get("target") or "").split():
            ref_id = target.lstrip("#")
            if ref_id and ref_id not in ids:
                ids.append(ref_id)
    return ids


def paragraph_text(p: _Element) -> str:
    sentences = p.findall("tei:s", NS)
    if sentences:
        return normalize_ws(" ".join(body_text(s) for s in sentences))
    return body_text(p)


def formula_text(formula: _Element) -> str:
    """Formula text with its equation label separated: 'softmax(...)V (1)'."""
    body = element_text(formula, skip=lambda e: e.tag == _tag("label"))
    label = formula.find("tei:label", NS)
    label_text = element_text(label) if label is not None else ""
    return normalize_ws(f"{body} {label_text}")


def paragraph_citations(p: _Element, heading: str) -> list[InTextCitation]:
    """One citation record per citing sentence (or per paragraph when there are no <s>)."""
    sentences = p.findall("tei:s", NS)
    units = sentences if sentences else [p]
    out: list[InTextCitation] = []
    for unit in units:
        ids = _bibr_targets(unit)
        if ids:
            out.append(
                InTextCitation(section_heading=heading, sentence=body_text(unit), ref_ids=ids)
            )
    if sentences:
        # bibr refs sitting directly in <p>, outside any <s>
        loose = [r for r in p.findall("tei:ref", NS) if r.get("type") == "bibr"]
        ids = [t.lstrip("#") for r in loose for t in (r.get("target") or "").split()]
        if ids:
            out.append(
                InTextCitation(section_heading=heading, sentence=paragraph_text(p), ref_ids=ids)
            )
    return out


def div_content(div: _Element, heading: str) -> tuple[list[str], list[InTextCitation]]:
    """Paragraphs (with formulas folded into the preceding paragraph, list items as their own
    paragraphs) and citations from the direct children of a <div>."""
    paragraphs: list[str] = []
    citations: list[InTextCitation] = []
    for child in div:
        if child.tag == _tag("p"):
            text = paragraph_text(child)
            if text:
                paragraphs.append(text)
            citations.extend(paragraph_citations(child, heading))
        elif child.tag == _tag("formula"):
            text = formula_text(child)
            if not text:
                continue
            if paragraphs:
                paragraphs[-1] = f"{paragraphs[-1]} {text}"
            else:
                paragraphs.append(text)
        elif child.tag == _tag("list"):
            for item in child.iter(_tag("item")):
                text = paragraph_text(item)
                if text:
                    paragraphs.append(text)
                citations.extend(paragraph_citations(item, heading))
    return paragraphs, citations


def _content_divs(root: _Element) -> Iterable[tuple[_Element, bool]]:
    """Body and back divs holding their own content, in document order; skips bibliography."""
    text = root.find("tei:text", NS)
    if text is None:
        return
    for part_name, in_back in (("body", False), ("back", True)):
        part = text.find(f"tei:{part_name}", NS)
        if part is None:
            continue
        for div in part.iter(_tag("div")):
            if any(a.get("type") == "references" for a in div.iterancestors(_tag("div"))):
                continue
            if div.get("type") == "references":
                continue
            if div.getparent() is not None and div.getparent().tag == _tag("figDesc"):
                continue
            has_own = any(
                c.tag in (_tag("head"), _tag("p"), _tag("formula"), _tag("list")) for c in div
            )
            if has_own:
                yield div, in_back


def extract_sections(root: _Element) -> _SectionBuild:
    build = _SectionBuild()
    for div, in_back in _content_divs(root):
        head = div.find("tei:head", NS)
        heading_text = element_text(head) if head is not None else ""
        pairs = (
            split_heading_number(heading_text, head.get("n"), in_back)
            if head is not None
            else [("", None)]
        )
        final_heading = pairs[-1][0]
        paragraphs, citations = div_content(div, final_heading)
        if head is None and build.sections:
            # headless div = continuation of the previous section (GROBID splits on figures)
            prev = build.sections[-1]
            prev.paragraphs.extend(paragraphs)
            for c in citations:
                c.section_heading = prev.heading
            build.citations.extend(citations)
            continue
        for heading, number in pairs[:-1]:
            build.sections.append(
                Section(heading=heading, number=number, level=level_from_number(number))
            )
        heading, number = pairs[-1]
        build.sections.append(
            Section(
                heading=heading,
                number=number,
                level=level_from_number(number),
                paragraphs=paragraphs,
            )
        )
        build.citations.extend(citations)
    return build


# --------------------------------------------------------------------------- figures/tables


def _figure_label(fig: _Element, is_table: bool) -> str:
    label = fig.find("tei:label", NS)
    if label is not None and (text := element_text(label)):
        return f"{'Table' if is_table else 'Figure'} {text}"
    head = fig.find("tei:head", NS)
    if head is not None and (text := element_text(head).rstrip(":").strip()):
        return text
    return fig.get(XML_ID) or ("table" if is_table else "figure")


def _caption(fig: _Element) -> str:
    desc = fig.find("tei:figDesc", NS)
    if desc is None:
        return ""
    return _CAPTION_LABEL_PREFIX.sub("", body_text(desc)).strip()


def table_rows(table: _Element) -> list[list[str]]:
    """Cell texts row by row; a cell spanning k columns is padded with k-1 empty cells."""
    rows: list[list[str]] = []
    for row in table.findall("tei:row", NS):
        cells: list[str] = []
        for cell in row.findall("tei:cell", NS):
            cells.append(element_text(cell))
            try:
                span = int(cell.get("cols") or 1)
            except ValueError:
                span = 1
            cells.extend([""] * max(span - 1, 0))
        rows.append(cells)
    return rows


def extract_figures_and_tables(root: _Element) -> tuple[list[Figure], list[Table]]:
    figures: list[Figure] = []
    tables: list[Table] = []
    text = root.find("tei:text", NS)
    if text is None:
        return figures, tables
    for fig in text.iter(_tag("figure")):
        if fig.getparent() is not None and fig.getparent().tag == _tag("figure"):
            continue
        is_table = fig.get("type") == "table"
        label = _figure_label(fig, is_table)
        caption = _caption(fig)
        if not is_table:
            figures.append(Figure(label=label, caption=caption))
            continue
        table_el = fig.find("tei:table", NS)
        rows = table_rows(table_el) if table_el is not None else []
        rows = [r for r in rows if any(c for c in r)]
        raw = element_text(table_el) if table_el is not None and not rows else None
        tables.append(Table(label=label, caption=caption, rows=rows or None, raw_text=raw or None))
    return figures, tables


# --------------------------------------------------------------------------- references


def _clean_arxiv(value: str) -> str | None:
    m = _ARXIV_ID.search(value)
    return m.group(1) if m else None


def reference_from_bibl(bibl: _Element, ref_id: str) -> Reference:
    analytic = bibl.find("tei:analytic", NS)
    monogr = bibl.find("tei:monogr", NS)

    def title_of(el: _Element | None) -> str | None:
        if el is None:
            return None
        t = el.find("tei:title[@type='main']", NS)
        if t is None:
            t = el.find("tei:title", NS)
        text = element_text(t) if t is not None else ""
        return text or None

    analytic_title = title_of(analytic)
    monogr_title = title_of(monogr)
    title = analytic_title or monogr_title
    venue = monogr_title if analytic_title else None

    author_parent = (
        analytic if analytic is not None and analytic.find("tei:author", NS) is not None else monogr
    )
    authors: list[str] = []
    if author_parent is not None:
        for author in author_parent.findall("tei:author", NS):
            pers = author.find("tei:persName", NS)
            name = person_name(pers) if pers is not None else element_text(author)
            if name:
                authors.append(name)

    year: int | None = None
    for date in bibl.iter(_tag("date")):
        m = _YEAR.match(date.get("when") or "")
        if m:
            year = int(m.group(1))
            break

    doi = arxiv_id = None
    for idno in bibl.iter(_tag("idno")):
        kind = (idno.get("type") or "").lower()
        value = element_text(idno)
        if kind == "doi" and not doi:
            doi = value or None
        elif kind == "arxiv" and not arxiv_id:
            arxiv_id = _clean_arxiv(value)
        elif not kind and not arxiv_id and (m := _ARXIV_IN_TEXT.search(value)):
            arxiv_id = m.group(1)

    raw_note = bibl.find("tei:note[@type='raw_reference']", NS)
    raw = element_text(raw_note) if raw_note is not None else ""
    if not raw:
        raw = element_text(bibl)
    if not arxiv_id and (m := _ARXIV_IN_TEXT.search(raw)):
        arxiv_id = m.group(1)

    ptr = bibl.find(".//tei:ptr[@target]", NS)
    url = ptr.get("target") if ptr is not None else None

    return Reference(
        ref_id=ref_id,
        raw_string=raw,
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        doi=doi,
        arxiv_id=arxiv_id,
        url=url,
    )


def extract_references(root: _Element) -> list[Reference]:
    refs: list[Reference] = []
    for i, bibl in enumerate(root.iterfind(".//tei:back//tei:listBibl/tei:biblStruct", NS)):
        refs.append(reference_from_bibl(bibl, bibl.get(XML_ID) or f"b{i}"))
    return refs


# --------------------------------------------------------------------------- assembly


def _safe[T](fn: Callable[[], T], default: T, what: str, warnings: list[str]) -> T:
    try:
        return fn()
    except Exception as exc:  # never crash on partial/odd TEI; record it instead
        logger.warning("TEI %s extraction failed: %r", what, exc)
        warnings.append(f"{what} extraction failed: {exc!r}")
        return default


def _parse_xml(xml: str) -> _Element | None:
    parser = etree.XMLParser(recover=True, huge_tree=True, remove_blank_text=False)
    try:
        return etree.fromstring(xml.encode("utf-8"), parser)
    except etree.XMLSyntaxError:
        return None


def build_report(
    sections: list[Section],
    figures: list[Figure],
    tables: list[Table],
    references: list[Reference],
    citations: list[InTextCitation],
    abstract: str,
    title: str,
    warnings: list[str],
) -> ParseReport:
    known = {r.ref_id for r in references}
    resolved = unresolved = 0
    dangling: list[str] = []
    for c in citations:
        for rid in c.ref_ids:
            if rid in known:
                resolved += 1
            else:
                unresolved += 1
                if rid not in dangling:
                    dangling.append(rid)
    warnings = list(warnings)
    if not title:
        warnings.append("missing title")
    if not abstract:
        warnings.append("empty abstract")
    if not sections:
        warnings.append("no sections")
    if not references:
        warnings.append("no references")
    if dangling:
        shown = ", ".join(dangling[:_UNRESOLVED_WARNING_LIMIT])
        more = "" if len(dangling) <= _UNRESOLVED_WARNING_LIMIT else ", ..."
        warnings.append(f"{len(dangling)} unresolved citation target(s): {shown}{more}")
    return ParseReport(
        n_sections=len(sections),
        n_paragraphs=sum(len(s.paragraphs) for s in sections),
        n_figures=len(figures),
        n_tables=len(tables),
        n_references=len(references),
        n_in_text_citations=len(citations),
        n_resolved_citations=resolved,
        n_unresolved_citations=unresolved,
        warnings=warnings,
    )


def parse_tei(
    xml: str, paper_id: str, source_path: str, pdf_sha256: str | None = None
) -> ParsedPaper:
    """Convert GROBID TEI XML into a ParsedPaper. Never raises on partial TEI."""
    warnings: list[str] = []
    root = _parse_xml(xml)
    if root is None:
        warnings.append("TEI could not be parsed as XML")
        return ParsedPaper(
            paper_id=paper_id,
            source_path=source_path,
            pdf_sha256=pdf_sha256,
            title="",
            parse_report=build_report([], [], [], [], [], "", "", warnings),
        )

    title = _safe(lambda: extract_title(root), "", "title", warnings)
    authors = _safe(lambda: extract_authors(root), [], "authors", warnings)
    abstract = _safe(lambda: extract_abstract(root), "", "abstract", warnings)
    build = _safe(lambda: extract_sections(root), _SectionBuild(), "sections", warnings)
    figures, tables = _safe(
        lambda: extract_figures_and_tables(root), ([], []), "figures/tables", warnings
    )
    references = _safe(lambda: extract_references(root), [], "references", warnings)

    return ParsedPaper(
        paper_id=paper_id,
        source_path=source_path,
        pdf_sha256=pdf_sha256,
        title=title,
        authors=authors,
        abstract=abstract,
        sections=build.sections,
        figures=figures,
        tables=tables,
        references=references,
        in_text_citations=build.citations,
        parse_report=build_report(
            build.sections, figures, tables, references, build.citations, abstract, title, warnings
        ),
    )


# --------------------------------------------------------------------------- page attribution

UNATTRIBUTED_PAGE = 0
"""Key in `page_texts` output for text whose element carries no usable coordinates."""

# Elements attributed as a whole (their text is taken once; nested coords are not descended into).
_ATOMIC_UNITS = frozenset(
    _tag(t) for t in ("s", "head", "formula", "figure", "note", "biblStruct", "item")
)


def parse_coords(coords: str | None) -> list[int]:
    """1-based page numbers of the boxes in a GROBID `coords` attribute, one entry per box.

    Format: 'page,x,y,w,h;page,x,y,w,h;...'. Malformed boxes are skipped.
    """
    pages: list[int] = []
    for box in (coords or "").split(";"):
        head = box.strip().split(",", 1)[0].strip()
        if head.isdigit() and int(head) >= 1:
            pages.append(int(head))
    return pages


def _nearest_space(text: str, i: int) -> int:
    """Index of the whitespace closest to `i` (a cut there splits no word); `i` if none."""
    forward = next((j for j in range(i, len(text)) if text[j].isspace()), len(text))
    backward = next((j for j in range(i, -1, -1) if j < len(text) and text[j].isspace()), 0)
    return backward if i - backward <= forward - i else forward


def split_by_boxes(text: str, box_pages: list[int]) -> list[tuple[int, str]]:
    """Split `text` across the pages of its coordinate boxes.

    Each page gets a share of the characters proportional to its number of boxes (boxes are
    roughly lines/fragments, so this approximates where the text breaks). Pages are taken in
    order of first appearance; cut points are moved to the nearest whitespace so words are
    never split. No boxes → [] (caller decides where unattributed text goes).
    """
    if not box_pages or not text:
        return []
    counts: dict[int, int] = {}
    for page in box_pages:
        counts[page] = counts.get(page, 0) + 1
    if len(counts) == 1:
        return [(box_pages[0], text)]
    total = len(box_pages)
    out: list[tuple[int, str]] = []
    start = 0
    seen = 0
    pages = list(counts)
    for i, page in enumerate(pages):
        seen += counts[page]
        if i == len(pages) - 1:
            end = len(text)
        else:
            end = max(_nearest_space(text, round(len(text) * seen / total)), start)
        piece = text[start:end].strip()
        if piece:
            out.append((page, piece))
        start = end
    return out


def _bibl_text(bibl: _Element) -> str:
    """The text the parse keeps for a bibliography entry: the raw reference string."""
    raw_note = bibl.find("tei:note[@type='raw_reference']", NS)
    raw = element_text(raw_note) if raw_note is not None else ""
    return raw or element_text(bibl)


def _figure_text(fig: _Element) -> str:
    """Figure/table label, caption and table cells (what the parse keeps), without graphics."""
    label = fig.find("tei:label", NS)
    if label is None or not element_text(label):
        label = fig.find("tei:head", NS)
    desc = fig.find("tei:figDesc", NS)
    table = fig.find("tei:table", NS)
    parts = [
        element_text(label) if label is not None else "",
        body_text(desc) if desc is not None else "",
        element_text(table) if table is not None else "",
    ]
    return normalize_ws(" ".join(p for p in parts if p))


def _unit_text(el: _Element) -> str:
    if el.tag == _tag("formula"):
        return formula_text(el)
    if el.tag == _tag("figure"):
        return _figure_text(el)
    if el.tag == _tag("biblStruct"):
        return _bibl_text(el)
    if el.tag == _tag("s"):
        return body_text(el)
    if el.tag == _tag("persName"):
        return person_name(el)
    if el.tag == _tag("affiliation"):
        return affiliation_text(el) or ""
    return element_text(el)


def _text_units(el: _Element) -> Iterable[_Element]:
    """Non-overlapping text-bearing elements under `el`, in document order.

    Atomic units (sentences, heads, formulas, figures, notes, bibliography entries, list items)
    are yielded whole and never descended into, so e.g. a <ref> inside an <s>, or a <title>
    inside a <biblStruct>, is not counted twice. A <p> is yielded whole only when it contains
    no <s> (unsegmented text); otherwise its sentences are the units.
    """
    for child in el:
        if not isinstance(child.tag, str):
            continue
        unsegmented_p = child.tag == _tag("p") and child.find(".//tei:s", NS) is None
        if child.tag in _ATOMIC_UNITS or unsegmented_p:
            yield child
        else:
            yield from _text_units(child)


def _header_units(root: _Element) -> Iterable[_Element]:
    """Header elements the parse keeps: main title, author names and affiliations, abstract."""
    title_stmt = root.find(".//tei:teiHeader/tei:fileDesc/tei:titleStmt", NS)
    if title_stmt is not None:
        title = _first(title_stmt, "tei:title[@type='main']")
        if title is None:
            title = _first(title_stmt, "tei:title")
        if title is not None:
            yield title
    analytic = root.find(
        ".//tei:teiHeader/tei:fileDesc/tei:sourceDesc/tei:biblStruct/tei:analytic", NS
    )
    if analytic is not None:
        for author in analytic.findall("tei:author", NS):
            for part in ("tei:persName", "tei:affiliation"):
                found = author.find(part, NS)
                if found is not None:
                    yield found
    abstract = root.find(".//tei:teiHeader/tei:profileDesc/tei:abstract", NS)
    if abstract is not None:
        yield from _text_units(abstract)


def _body_units(root: _Element) -> Iterable[_Element]:
    text = root.find("tei:text", NS)
    if text is None:
        return
    for part_name in ("body", "back"):
        part = text.find(f"tei:{part_name}", NS)
        if part is not None:
            yield from _text_units(part)


def _attribute(el: _Element, text: str, fallback_page: int, out: dict[int, list[str]]) -> None:
    pieces = split_by_boxes(text, parse_coords(el.get("coords")))
    if not pieces:
        pieces = [(fallback_page, text)]
    for page, piece in pieces:
        out.setdefault(page, []).append(piece)


def page_texts(xml: str) -> dict[int, str]:
    """Text recovered by the parse, attributed to 1-based PDF pages via GROBID `coords`.

    Requires TEI produced with `teiCoordinates` (see grobid_client.TEI_COORDINATES). Each
    text-bearing element (sentence, head, formula, figure/table incl. caption and cells, note,
    bibliography entry, header title/author/affiliation, unsegmented paragraph) is counted once
    — nested elements are never double counted — and assigned to the page(s) of its coordinate
    boxes. An element whose boxes span several pages has its characters split proportionally to
    the number of boxes on each page (see `split_by_boxes`).

    Elements without coordinates go to page 1 if they are header elements (title, authors,
    abstract) and otherwise to `UNATTRIBUTED_PAGE` (0), which callers should report separately.
    Returns {page: text} with one line per element, pages in ascending order; empty on bad XML.
    """
    root = _parse_xml(xml)
    if root is None:
        return {}
    out: dict[int, list[str]] = {}
    for units, fallback_page in ((_header_units(root), 1), (_body_units(root), UNATTRIBUTED_PAGE)):
        for el in units:
            text = _unit_text(el)
            if text:
                _attribute(el, text, fallback_page, out)
    return {page: "\n".join(out[page]) for page in sorted(out)}
