"""Verbatim text from PDFs: line joining, anchored spans, and two-column table recovery.

Nothing here rewrites text. The only transformations are layout ones: a PDF line break becomes
one space (or nothing, when a word was split across lines), and whitespace runs collapse.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pdfplumber

DICT_PATH = Path("/usr/share/dict/words")
ROW_GAP_PX = 20.0
"""Vertical gap between consecutive table lines that starts a new row (lines are ~16px apart
inside a row and ~26px apart between rows in the documents we handle)."""
EDGE_TOLERANCE_PX = 8.0


@dataclass(frozen=True)
class Word:
    page: int
    text: str
    x0: float
    x1: float
    top: float


@dataclass(frozen=True)
class Line:
    page: int
    top: float
    words: tuple[Word, ...]

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


class SpanError(ValueError):
    """An anchor was not found exactly once in the source text."""


def read_words(pdf_path: Path) -> list[Word]:
    words: list[Word] = []
    with pdfplumber.open(pdf_path) as pdf:
        for number, page in enumerate(pdf.pages, start=1):
            for w in page.extract_words():
                words.append(Word(number, w["text"], w["x0"], w["x1"], w["top"]))
    return words


def group_lines(words: list[Word], tolerance: float = 2.0) -> list[Line]:
    lines: list[Line] = []
    for w in sorted(words, key=lambda w: (w.page, round(w.top), w.x0)):
        if lines and lines[-1].page == w.page and abs(lines[-1].top - w.top) <= tolerance:
            last = lines[-1]
            lines[-1] = Line(last.page, last.top, (*last.words, w))
        else:
            lines.append(Line(w.page, w.top, (w,)))
    return [Line(ln.page, ln.top, tuple(sorted(ln.words, key=lambda w: w.x0))) for ln in lines]


def join_lines(lines: list[str]) -> str:
    """Join PDF lines with single spaces; a line ending in 'letter-' joins without a space
    so hyphenated compounds split across lines stay intact."""
    out = ""
    for line in (ln.strip() for ln in lines):
        if not line:
            continue
        if not out:
            out = line
        elif re.search(r"[A-Za-z]-$", out):
            out += line
        else:
            out += " " + line
    return " ".join(out.split())


def full_text(pdf_path: Path) -> str:
    return join_lines([ln.text for ln in group_lines(read_words(pdf_path))])


def span(text: str, start: str, end: str) -> str:
    """The exact text from the unique occurrence of `start` through the first `end` after it."""
    count = text.count(start)
    if count != 1:
        raise SpanError(f"start anchor found {count} times: {start[:60]!r}")
    i = text.index(start)
    j = text.find(end, i)
    if j < 0:
        raise SpanError(f"end anchor not found after start: {end[:60]!r}")
    return text[i : j + len(end)]


@lru_cache(maxsize=1)
def _dictionary() -> frozenset[str]:
    if not DICT_PATH.exists():
        return frozenset()
    return frozenset(w.strip().lower() for w in DICT_PATH.read_text().splitlines())


def _is_word(token: str) -> bool:
    return token.lower().strip(".,;:()") in _dictionary()


def joins_mid_word(prev: str, nxt: str, prev_reaches_edge: bool) -> bool:
    """Whether a narrow table cell wrapped inside a word (join with no space).

    True when the concatenation is a dictionary word and either fragment is not a word, or the
    first fragment reaches the column's right edge (both 'empiric' and 'al' are words, but
    'empiric' fills the column, so 'empirical' was wrapped)."""
    head = prev.rsplit("-", 1)[-1]
    joined = head + nxt
    if not nxt or not nxt[0].islower() or not _is_word(joined):
        return False
    return not _is_word(head) or not _is_word(nxt) or prev_reaches_edge


@dataclass(frozen=True)
class TableRow:
    left: str
    right: str


def two_column_rows(lines: list[Line], split_x: float) -> list[TableRow]:
    """Rows of a borderless two-column table. A new row starts on a page change or a vertical
    gap > ROW_GAP_PX. Left-column fragments are joined with the mid-word rule above; the
    right column is joined as ordinary text."""
    left_parts: list[tuple[str, float]] = []
    for ln in lines:
        left = [w for w in ln.words if w.x0 < split_x]
        if left:
            left_parts.append((" ".join(w.text for w in left), max(w.x1 for w in left)))
    col_edge = max((x1 for _, x1 in left_parts), default=0.0)

    rows: list[TableRow] = []
    cur_left: list[tuple[str, float]] = []
    cur_right: list[str] = []
    prev: Line | None = None

    def flush() -> None:
        if cur_left or cur_right:
            text = ""
            for i, (frag, _x1) in enumerate(cur_left):
                if i and joins_mid_word(
                    cur_left[i - 1][0], frag, cur_left[i - 1][1] >= col_edge - EDGE_TOLERANCE_PX
                ):
                    text += frag
                else:
                    text += (" " if text else "") + frag
            rows.append(TableRow(" ".join(text.split()), join_lines(cur_right)))

    for ln in lines:
        new_row = prev is not None and (ln.page != prev.page or ln.top - prev.top > ROW_GAP_PX)
        if new_row:
            flush()
            cur_left, cur_right = [], []
        left = [w for w in ln.words if w.x0 < split_x]
        right = [w for w in ln.words if w.x0 >= split_x]
        if left:
            cur_left.append((" ".join(w.text for w in left), max(w.x1 for w in left)))
        if right:
            cur_right.append(" ".join(w.text for w in right))
        prev = ln
    flush()
    return rows


def table_lines(
    lines: list[Line], pages: list[int], header: str, end_marker: str | None
) -> list[Line]:
    """Lines after the `header` line on the first page, through the given pages, stopping
    before the first line that starts with `end_marker`."""
    selected: list[Line] = []
    started = False
    for ln in lines:
        if ln.page not in pages:
            continue
        if not started:
            if ln.page == pages[0] and ln.text.startswith(header):
                started = True
            continue
        if end_marker and ln.text.startswith(end_marker):
            break
        selected.append(ln)
    if not started:
        raise SpanError(f"table header not found on page {pages[0]}: {header!r}")
    return selected
