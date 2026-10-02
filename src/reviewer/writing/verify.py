"""Turn the model's WritingOutput into verified findings. Pure: no I/O, no LLM.

Every quote must occur verbatim (after whitespace, quote, and dash normalization) in the part
of the paper it claims to come from; anything that fails is dropped with a reason. Where a
check has an order (a term used before its definition), code decides the order from positions
in the text, not the model.
"""

import re
import unicodedata
from dataclasses import dataclass, field

from reviewer.claims.prompt_builder import (
    full_text,
    intro_indices,
    matching_subtrees,
    row_line,
)
from reviewer.models import (
    AbstractClaimTrace,
    AbstractClaimVerdict,
    CaptionCheck,
    ContributionCheck,
    ContributionVerdict,
    DroppedFinding,
    ParsedPaper,
    RelatedWorkVerdict,
    TermUse,
    TermVerdict,
    WritingCheck,
    WritingFinding,
    WritingOutput,
)

REFERENCES_RE = re.compile(r"\b(references|bibliography)\b", re.IGNORECASE)
"""Headings left out of the writing context (with their subsections)."""
RELATED_WORK_RE = re.compile(
    r"\b(related\s+works?|prior\s+work|previous\s+work|literature|background)\b", re.IGNORECASE
)
MIN_SPAN_WORDS = 5
MIN_PIECE_WORDS = 3
"""Each piece of a span the model elided with an ellipsis must be at least this long."""
PIECE_WINDOW = 600
"""Elided pieces must follow each other within this many characters."""
MIN_TERM_CHARS = 2
SNIPPET_CHARS = 240

_TRANSLATE = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "−": "-",
        "‐": "-",
        "‑": "-",
        "­": None,
        "​": None,
    }
)
_WS = re.compile(r"\s+")
_ELLIPSIS = re.compile(r"\s*(?:\[\s*(?:\.\s*){3}\]|\.\s*\.\s*\.|…)\s*")
_EDGE = " \t\n\"'.,;:()[]"


def norm(text: str) -> str:
    """NFKC (ligatures), straight quotes, ASCII hyphens, single spaces."""
    return _WS.sub(" ", unicodedata.normalize("NFKC", text).translate(_TRANSLATE)).strip()


def _fold(text: str) -> str:
    low = text.lower()
    return low if len(low) == len(text) else text


def n_words(text: str) -> int:
    return len(text.split())


@dataclass(frozen=True)
class Block:
    """A stretch of the paper text the model saw, normalized."""

    kind: str
    """'abstract', 'section', 'table', 'figure', or 'caption' (a table caption)."""
    label: str
    """Section heading or figure/table label."""
    text: str
    start: int
    """Offset of this block in PaperText.flat."""


@dataclass
class PaperText:
    """Every block in reading order, plus one flat string to search and order positions in."""

    blocks: list[Block] = field(default_factory=list)
    flat: str = ""
    folded: str = ""
    intro: set[int] = field(default_factory=set)
    """Block indices of the introduction."""
    related: set[int] = field(default_factory=set)
    """Block indices of related-work and background sections."""

    def add(self, kind: str, label: str, text: str) -> int:
        text = norm(text)
        if self.flat:
            self.flat += "\n"
        block = Block(kind, label, text, len(self.flat))
        self.flat += text
        self.blocks.append(block)
        return len(self.blocks) - 1

    def block_at(self, pos: int) -> int:
        idx = 0
        for i, b in enumerate(self.blocks):
            if b.start <= pos:
                idx = i
        return idx


def paper_text(paper: ParsedPaper) -> PaperText:
    """The same material as the writing context (judge_v2 minus reference sections, plus
    figure captions), as blocks."""
    pt = PaperText()
    pt.add("abstract", "abstract", paper.abstract)
    secs = paper.sections
    skipped = set(matching_subtrees(secs, REFERENCES_RE))
    intro = set(intro_indices(secs))
    # GROBID sometimes splits the introduction at a running header or footer, leaving its
    # tail under an unnumbered heading; those sections still belong to the introduction.
    nxt = max(intro, default=-1) + 1
    while intro and nxt < len(secs) and not secs[nxt].number:
        intro.add(nxt)
        nxt += 1
    related = set(matching_subtrees(secs, RELATED_WORK_RE))
    for i, sec in enumerate(secs):
        st = full_text(sec)
        if i in skipped or not st.text:
            continue
        b = pt.add("section", st.heading or "(untitled section)", st.text)
        if i in intro:
            pt.intro.add(b)
        if i in related:
            pt.related.add(b)
    for t in paper.tables:
        pt.add("caption", t.label, t.caption)
        rows = [line for line in (row_line(r) for r in t.rows or []) if line]
        pt.add("table", t.label, "\n".join(rows) or t.raw_text or "")
    for f in paper.figures:
        pt.add("figure", f.label, f.caption)
    pt.folded = _fold(pt.flat)
    return pt


@dataclass(frozen=True)
class Hit:
    start: int
    end: int
    block: int


def find_span(pt: PaperText, span: str, allowed: set[int] | None = None) -> Hit | None:
    """First occurrence of `span` inside one allowed block (all blocks when None). An ellipsis
    splits the span into pieces that must each be found, in order, close together."""
    pieces = [p.strip(_EDGE) for p in _ELLIPSIS.split(norm(span))]
    pieces = [_fold(p) for p in pieces if p]
    if not pieces or (len(pieces) > 1 and min(n_words(p) for p in pieces) < MIN_PIECE_WORDS):
        return None
    hay = pt.folded
    pos = 0
    while True:
        start = hay.find(pieces[0], pos)
        if start < 0:
            return None
        end = start + len(pieces[0])
        ok = True
        for piece in pieces[1:]:
            nxt = hay.find(piece, end, end + PIECE_WINDOW + len(piece))
            if nxt < 0:
                ok = False
                break
            end = nxt + len(piece)
        b = pt.block_at(start)
        blk = pt.blocks[b]
        inside = end <= blk.start + len(blk.text)
        if ok and inside and (allowed is None or b in allowed):
            return Hit(start, end, b)
        pos = start + 1


def blocks_of(pt: PaperText, *kinds: str) -> set[int]:
    return {i for i, b in enumerate(pt.blocks) if b.kind in kinds}


def location(pt: PaperText, hit: Hit) -> str:
    b = pt.blocks[hit.block]
    return b.kind if b.kind == "abstract" else b.label


def snippet(pt: PaperText, start: int, end: int) -> str:
    """The sentence(s) around [start, end), cut to the enclosing block, verbatim from the
    normalized text."""
    blk = pt.blocks[pt.block_at(start)]
    lo = max(blk.start, start - SNIPPET_CHARS // 2)
    hi = min(blk.start + len(blk.text), end + SNIPPET_CHARS // 2)
    left = pt.flat.rfind(". ", lo, start)
    lo = left + 2 if left >= 0 else lo
    right = pt.flat.find(". ", end, hi)
    hi = right + 1 if right >= 0 else hi
    return pt.flat[lo:hi].strip()


def span_text(pt: PaperText, hit: Hit) -> str:
    return pt.flat[hit.start : hit.end]


# --------------------------------------------------------------------------- checks


@dataclass
class Verified:
    findings: list[WritingFinding] = field(default_factory=list)
    dropped: list[DroppedFinding] = field(default_factory=list)
    trace: list[AbstractClaimTrace] = field(default_factory=list)
    contribution: ContributionCheck | None = None
    related_work: RelatedWorkVerdict | None = None

    def drop(self, check: WritingCheck, reason: str, item: object) -> None:
        dump = item.model_dump_json() if hasattr(item, "model_dump_json") else str(item)
        self.dropped.append(DroppedFinding(check=check, reason=reason, item=dump))


def _short(span: str) -> bool:
    return n_words(span) < MIN_SPAN_WORDS


def verify_abstract(pt: PaperText, items: list[AbstractClaimTrace], out: Verified) -> None:
    check = WritingCheck.abstract_overclaim
    abstract = blocks_of(pt, "abstract")
    body = set(range(len(pt.blocks))) - abstract
    for item in items:
        a = find_span(pt, item.abstract_span, abstract)
        if a is None:
            out.drop(check, "abstract_span not found verbatim in the abstract", item)
            continue
        r = find_span(pt, item.result_span, body) if item.result_span.strip() else None
        v = item.verdict
        if item.result_span.strip() and r is None:
            out.drop(check, "result_span not found verbatim outside the abstract", item)
            continue
        if v == AbstractClaimVerdict.supported and r is None:
            out.drop(check, "supported without a result_span", item)
            continue
        if v == AbstractClaimVerdict.overstated and (r is None or not item.gap.strip()):
            out.drop(check, "overstated needs a verified result_span and a gap", item)
            continue
        if v == AbstractClaimVerdict.not_found and r is not None:
            out.drop(check, "not_found but a result_span was quoted", item)
            continue
        out.trace.append(item)
        if v not in (AbstractClaimVerdict.overstated, AbstractClaimVerdict.not_found):
            continue
        if _short(item.abstract_span):
            out.drop(check, f"abstract_span under {MIN_SPAN_WORDS} words", item)
            continue
        evidence = [f"{span_text(pt, r)} [{location(pt, r)}]"] if r else []
        detail = item.gap.strip() or "No result in the text shown bears on this claim."
        out.findings.append(
            WritingFinding(
                check=check,
                quote=span_text(pt, a),
                evidence=evidence,
                location="abstract",
                detail=f"{v.value}: {detail}",
            )
        )


def verify_contribution(pt: PaperText, c: ContributionCheck, out: Verified) -> None:
    check = WritingCheck.unclear_contribution
    hit = find_span(pt, c.intro_span, pt.intro or blocks_of(pt, "section"))
    if hit is None:
        out.drop(check, "intro_span not found verbatim in the introduction", c)
        return
    out.contribution = c
    if c.verdict == ContributionVerdict.clear:
        return
    if not c.problem.strip() or _short(c.intro_span):
        out.drop(check, "not clear, but no problem stated or span too short", c)
        return
    out.findings.append(
        WritingFinding(
            check=check,
            quote=span_text(pt, hit),
            location=location(pt, hit),
            detail=f"{c.verdict.value}: {c.problem.strip()}",
        )
    )


def _term_re(term: str) -> re.Pattern[str]:
    return re.compile(r"(?<![\w-])" + re.escape(_fold(norm(term))) + r"(?![\w])")


def verify_term(pt: PaperText, t: TermUse, out: Verified) -> None:
    """Code decides first use and definition order: the earliest occurrence of the term in the
    section text is the first use, whatever span the model quoted."""
    check = WritingCheck.undefined_term
    term = norm(t.term).strip(_EDGE)
    if len(term) < MIN_TERM_CHARS:
        out.drop(check, "term too short to locate", t)
        return
    sections = blocks_of(pt, "section")
    hit = find_span(pt, t.first_use_span, sections)
    if hit is None:
        out.drop(check, "first_use_span not found verbatim in the section text", t)
        return
    pattern = _term_re(term)
    if not pattern.search(pt.folded, hit.start, hit.end):
        out.drop(check, "first_use_span does not contain the term", t)
        return
    first = next(
        (m for m in pattern.finditer(pt.folded) if pt.block_at(m.start()) in sections), None
    )
    assert first is not None  # the span itself contains it
    expansion = next(
        (
            m
            for m in re.finditer(r"\(\s*" + re.escape(_fold(term)) + r"s?\s*\)", pt.folded)
            if pt.block_at(m.start()) in sections
        ),
        None,
    )
    definition: Hit | None = None
    if t.definition_span.strip():
        definition = find_span(pt, t.definition_span, sections | blocks_of(pt, "figure", "caption"))
        if definition is None:
            out.drop(check, "definition_span not found verbatim", t)
            return
    elif expansion is not None:
        definition = Hit(expansion.start(), expansion.end(), pt.block_at(expansion.start()))
    if definition is not None and definition.start <= first.end():
        out.drop(check, "defined at or before its first use in the body", t)
        return
    evidence = []
    verdict = TermVerdict.never_defined
    if definition is not None:
        verdict = TermVerdict.defined_later
        evidence.append(
            f"{snippet(pt, definition.start, definition.end)} [{location(pt, definition)}]"
        )
    first_hit = Hit(first.start(), first.end(), pt.block_at(first.start()))
    detail = (
        f'"{term}" is first used in {location(pt, first_hit)} and defined later in '
        f"{location(pt, definition)}."
        if definition is not None
        else f'"{term}" is used without a definition in the text shown.'
    )
    out.findings.append(
        WritingFinding(
            check=check,
            quote=snippet(pt, first.start(), first.end()),
            evidence=evidence,
            location=location(pt, first_hit),
            detail=f"{verdict.value}: {detail}",
        )
    )


_LABEL_PREFIX = re.compile(r"^\s*(?:fig(?:ure)?|table)\.?\s*[A-Z]?\.?\d+\s*[:.]\s*", re.IGNORECASE)
"""The context shows captions as '<label>: <caption>'; a quote may carry that prefix."""
_LABEL_NUM = re.compile(r"(\d+|[A-Z]\.?\d+)\s*$")


def _label_kind(label: str) -> str | None:
    low = label.lower()
    if low.startswith("fig"):
        return "figure"
    if low.startswith("tab"):
        return "table"
    return None


def verify_caption(pt: PaperText, c: CaptionCheck, out: Verified) -> None:
    check = WritingCheck.caption_not_standalone
    kind = _label_kind(c.label)
    captions = blocks_of(pt, "figure", "caption")
    hit = find_span(pt, c.caption_span, captions) or find_span(
        pt, _LABEL_PREFIX.sub("", c.caption_span), captions
    )
    if hit is None:
        out.drop(check, "caption_span not found verbatim in any caption", c)
        return
    found = pt.blocks[hit.block]
    found_kind = _label_kind(found.label)
    if found_kind is None:
        out.drop(check, f"caption belongs to an unlabelled entry ({found.label})", c)
        return
    want = _LABEL_NUM.search(c.label.strip())
    have = _LABEL_NUM.search(found.label.strip())
    if kind != found_kind or not want or not have or want.group(1) != have.group(1):
        out.drop(check, f"caption_span is from {found.label}, not {c.label}", c)
        return
    if not c.missing.strip():
        out.drop(check, "nothing said to be missing", c)
        return
    out.findings.append(
        WritingFinding(
            check=check,
            quote=span_text(pt, hit),
            location=found.label,
            detail=c.missing.strip(),
        )
    )


def verify_related(pt: PaperText, output: WritingOutput, out: Verified) -> None:
    check = WritingCheck.related_work_list
    rw = output.related_work
    out.related_work = rw.verdict
    if rw.verdict != RelatedWorkVerdict.mostly_lists:
        return
    region = (pt.related | pt.intro) or blocks_of(pt, "section")
    hits = [h for s in rw.listing_spans if not _short(s) and (h := find_span(pt, s, region))]
    if not hits:
        out.drop(check, "no listing span found verbatim in related work or the introduction", rw)
        return
    detail = "Prior work is mostly listed without saying how this paper relates to it."
    if rw.engaging_span.strip() and (e := find_span(pt, rw.engaging_span, region)):
        detail += f' The one engaging sentence quoted: "{span_text(pt, e)}"'
    out.findings.append(
        WritingFinding(
            check=check,
            quote=span_text(pt, hits[0]),
            evidence=[f"{span_text(pt, h)} [{location(pt, h)}]" for h in hits[1:]],
            location=location(pt, hits[0]),
            detail=detail,
        )
    )


def verify_output(paper: ParsedPaper, output: WritingOutput) -> Verified:
    pt = paper_text(paper)
    out = Verified()
    verify_abstract(pt, output.abstract_claims, out)
    verify_contribution(pt, output.contribution, out)
    for t in output.undefined_terms:
        verify_term(pt, t, out)
    for c in output.captions:
        verify_caption(pt, c, out)
    verify_related(pt, output, out)
    return out
