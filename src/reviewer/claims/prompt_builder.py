"""Select the parts of a ParsedPaper the claim extractor sees, and render them as text.

Pure functions, no network. We send a selective context, never the whole paper: title,
abstract, contributions statement, full introduction and conclusion (with subsections), all
section headings, figure captions, tables (label, caption, and cell contents or GROBID's raw
text, capped per table), and the first paragraph of each method or experiment section.

Fallback sections (pdftotext text spliced in for pages the parse under-recovered) never count
as introduction, conclusion, or openers and are not listed among the headings; each is rendered
in its own <fallback_pages> block, capped per page, and only when there is any.
"""

import logging
import re

from pydantic import BaseModel, Field

from reviewer.config import Settings
from reviewer.models import ParsedPaper, Section, Table

logger = logging.getLogger(__name__)

INTRO_RE = re.compile(r"introduction", re.IGNORECASE)
CONCLUSION_RE = re.compile(
    r"\b(conclusions?|discussion|summary|concluding\s+remarks)\b", re.IGNORECASE
)
EXCLUDED_RE = re.compile(r"\b(related\s+work|background)\b", re.IGNORECASE)
METHOD_EXPERIMENT_RE = re.compile(
    r"\b(method|approach|model|architecture|framework|algorithm|experiment|evaluation|setup"
    r"|set-up|result|analys[ie]s|ablation)",
    re.IGNORECASE,
)
WHITESPACE_RE = re.compile(r"\s+")
RELATED_RE = re.compile(
    r"\b(related\s+work|prior\s+work|literature\s+review|references|bibliography)\b",
    re.IGNORECASE,
)
"""Sections left out of full-paper contexts (with their subsections)."""
UNCUT = 1_000_000

DEFAULT_TABLE_CHAR_BUDGET: int = Settings.model_fields["table_char_budget"].default
"""Per-table character cap when the caller passes none (the config default)."""
DEFAULT_FALLBACK_CHAR_BUDGET = 4000
"""Per-page character cap for fallback (pdftotext) text."""
NO_CELLS_NOTE = "(no cell content recovered)"
TRUNCATED_NOTE = "[… truncated]"


class SectionText(BaseModel):
    heading: str
    """Heading with its number, e.g. '3.2 Training setup'."""
    text: str


class TableText(BaseModel):
    header: str
    """Label and caption, e.g. 'Table 2: Main results.'."""
    body: str
    """Rows (' | '-joined cells, one per line), else GROBID's raw text, else a note saying no
    content was recovered; capped at the table budget with a note saying what was cut."""


class PromptContext(BaseModel):
    """The excerpts of a paper the extractor and critic see."""

    title: str
    abstract: str
    contributions: str | None = None
    introduction: list[SectionText] = Field(default_factory=list)
    conclusion: list[SectionText] = Field(default_factory=list)
    section_headings: list[str] = Field(default_factory=list)
    """Every heading with its number, indented two spaces per level below the top."""
    figure_captions: list[str] = Field(default_factory=list)
    tables: list[TableText] = Field(default_factory=list)
    method_experiment_openers: list[SectionText] = Field(default_factory=list)
    """First paragraph of each method or experiment section or subsection."""
    fallback_pages: list[SectionText] = Field(default_factory=list)
    """pdftotext text of pages the parse under-recovered, one entry per page, capped."""


# --------------------------------------------------------------------------- selection


def numbered_heading(section: Section) -> str:
    return f"{section.number} {section.heading}" if section.number else section.heading


def subtree(sections: list[Section], index: int) -> list[int]:
    """Indices of section `index` and every following section nested below it."""
    level = sections[index].level
    out = [index]
    for j in range(index + 1, len(sections)):
        if sections[j].level <= level:
            break
        out.append(j)
    return out


def intro_indices(sections: list[Section]) -> list[int]:
    """The first section whose heading mentions 'introduction' (else the first section),
    with its subsections."""
    if not sections:
        return []
    start = next((i for i, s in enumerate(sections) if INTRO_RE.search(s.heading)), 0)
    return subtree(sections, start)


def matching_subtrees(sections: list[Section], pattern: re.Pattern[str]) -> list[int]:
    """Indices of every section whose heading matches `pattern`, plus their subsections."""
    selected: set[int] = set()
    for i, s in enumerate(sections):
        if i not in selected and pattern.search(s.heading):
            selected.update(subtree(sections, i))
    return sorted(selected)


def conclusion_indices(sections: list[Section], exclude: set[int]) -> list[int]:
    return [i for i in matching_subtrees(sections, CONCLUSION_RE) if i not in exclude]


def opener_indices(sections: list[Section], exclude: set[int]) -> list[int]:
    """Sections whose heading names a method or experiment, plus every subsection under them
    (e.g. '2.2 Masking to Reduce Copying' under '2 Methods'), outside the intro, conclusion,
    related work, and background (and their subsections)."""
    blocked = exclude | set(matching_subtrees(sections, EXCLUDED_RE))
    return [i for i in matching_subtrees(sections, METHOD_EXPERIMENT_RE) if i not in blocked]


def first_paragraph(section: Section) -> str | None:
    return next((p.strip() for p in section.paragraphs if p.strip()), None)


def full_text(section: Section) -> SectionText:
    body = "\n\n".join(p.strip() for p in section.paragraphs if p.strip())
    return SectionText(heading=numbered_heading(section), text=body)


def heading_outline(sections: list[Section]) -> list[str]:
    return ["  " * max(s.level - 1, 0) + numbered_heading(s) for s in sections]


def collapse(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def row_line(row: list[str]) -> str:
    """Cells joined by ' | ' with whitespace collapsed and empty trailing cells dropped."""
    cells = [collapse(c) for c in row]
    while cells and not cells[-1]:
        cells.pop()
    return " | ".join(cells)


def cap_rows(lines: list[str], budget: int) -> str:
    """Keep whole rows while the body (rows joined by newlines) fits in `budget` characters;
    then say how many rows were cut."""
    kept: list[str] = []
    size = 0
    for line in lines:
        added = len(line) + (1 if kept else 0)
        if size + added > budget:
            break
        kept.append(line)
        size += added
    n_cut = len(lines) - len(kept)
    if n_cut:
        kept.append(f"[… {n_cut} more rows not shown]")
    return "\n".join(kept)


def cap_text(text: str, budget: int) -> str:
    """Cut raw text to `budget` characters, at the last whitespace if there is one."""
    if len(text) <= budget:
        return text
    cut = text[:budget]
    space = cut.rfind(" ")
    if space > budget // 2:
        cut = cut[:space]
    return f"{cut.rstrip()} {TRUNCATED_NOTE}"


def table_text(table: Table, budget: int) -> TableText:
    """Rows if GROBID recovered cell structure, else its raw text, else a note."""
    lines = [line for line in (row_line(r) for r in table.rows or []) if line]
    raw = collapse(table.raw_text or "")
    if lines:
        body = cap_rows(lines, budget)
    elif raw:
        body = cap_text(raw, budget)
    else:
        body = NO_CELLS_NOTE
    caption = collapse(table.caption)
    header = f"{table.label}: {caption}" if caption else table.label
    return TableText(header=header, body=body)


def fallback_text(section: Section, budget: int) -> SectionText | None:
    """A fallback section as 'Page N (...)' plus its text capped at `budget` characters;
    None when it has no text."""
    body = "\n\n".join(p.strip() for p in section.paragraphs if p.strip())
    if not body:
        return None
    page = section.fallback_page if section.fallback_page is not None else section.heading
    heading = f"Page {page} (pdftotext fallback, may be garbled or duplicate other text):"
    return SectionText(heading=heading, text=cap_text(body, budget))


def build_context(
    paper: ParsedPaper,
    table_char_budget: int = DEFAULT_TABLE_CHAR_BUDGET,
    fallback_char_budget: int = DEFAULT_FALLBACK_CHAR_BUDGET,
) -> PromptContext:
    """Select the excerpts. `table_char_budget` caps each table's body (see `table_text`);
    `fallback_char_budget` caps each fallback page's text."""
    sections = [s for s in paper.sections if not s.fallback]
    fallbacks = [
        text
        for text in (fallback_text(s, fallback_char_budget) for s in paper.sections if s.fallback)
        if text is not None
    ]
    intro = intro_indices(sections)
    conclusion = conclusion_indices(sections, exclude=set(intro))
    openers = opener_indices(sections, exclude=set(intro) | set(conclusion))

    opener_texts = []
    for i in openers:
        paragraph = first_paragraph(sections[i])
        if paragraph is not None:
            opener_texts.append(SectionText(heading=numbered_heading(sections[i]), text=paragraph))

    return PromptContext(
        title=paper.title,
        abstract=paper.abstract,
        contributions=paper.contributions,
        introduction=[full_text(sections[i]) for i in intro],
        conclusion=[full_text(sections[i]) for i in conclusion],
        section_headings=heading_outline(sections),
        figure_captions=[f"{f.label}: {f.caption}" for f in paper.figures],
        tables=[table_text(t, table_char_budget) for t in paper.tables],
        method_experiment_openers=opener_texts,
        fallback_pages=fallbacks,
    )


# --------------------------------------------------------------------------- rendering


def _tag(name: str, body: str) -> str:
    return f"<{name}>\n{body.strip() or '(none)'}\n</{name}>"


def _sections(items: list[SectionText]) -> str:
    return "\n\n".join(f"## {s.heading}\n{s.text}" if s.text else f"## {s.heading}" for s in items)


def _tables(items: list[TableText]) -> str:
    return "\n\n".join(f"### {t.header}\n{t.body}" for t in items)


def render(ctx: PromptContext) -> str:
    """Render the context as tagged blocks. Empty blocks say '(none)'."""
    tables = _tables(ctx.tables)
    parts = [
        _tag("title", ctx.title),
        _tag("abstract", ctx.abstract),
        _tag("contributions_statement", ctx.contributions or "(no contributions statement found)"),
        _tag("introduction", _sections(ctx.introduction)),
        _tag("conclusion", _sections(ctx.conclusion)),
        _tag("section_headings", "\n".join(ctx.section_headings)),
        _tag("figure_captions", "\n".join(ctx.figure_captions)),
        _tag("tables", tables),
        _tag("method_and_experiment_openers", _sections(ctx.method_experiment_openers)),
    ]
    if ctx.fallback_pages:
        parts.append(
            _tag(
                "fallback_pages", "\n\n".join(f"{s.heading}\n{s.text}" for s in ctx.fallback_pages)
            )
        )
    text = "\n\n".join(parts)
    logger.info(
        "rendered context for %r: %d chars (~%d tokens), of which %d tables: %d chars; "
        "%d fallback pages",
        ctx.title,
        len(text),
        len(text) // 4,
        len(ctx.tables),
        len(tables),
        len(ctx.fallback_pages),
    )
    return text


def render_full(paper: ParsedPaper) -> str:
    """The whole paper in the same tagged layout as `render`: every section except related work
    and references (with their subsections), every table and fallback page uncut."""
    sections = [s for s in paper.sections if not s.fallback]
    skipped = set(matching_subtrees(sections, RELATED_RE))
    body = [full_text(sec) for i, sec in enumerate(sections) if i not in skipped]
    fallbacks = [
        t for t in (fallback_text(s, UNCUT) for s in paper.sections if s.fallback) if t is not None
    ]
    parts = [
        _tag("title", paper.title),
        _tag("abstract", paper.abstract),
        _tag("sections", _sections(body)),
        _tag("figure_captions", "\n".join(f"{f.label}: {f.caption}" for f in paper.figures)),
        _tag("tables", _tables([table_text(t, UNCUT) for t in paper.tables])),
    ]
    if fallbacks:
        parts.append(
            _tag("fallback_pages", "\n\n".join(f"{s.heading}\n{s.text}" for s in fallbacks))
        )
    text = "\n\n".join(parts)
    logger.info(
        "rendered full paper %r: %d chars (~%d tokens)", paper.title, len(text), len(text) // 4
    )
    return text
