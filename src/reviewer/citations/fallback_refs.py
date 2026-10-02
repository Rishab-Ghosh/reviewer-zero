"""Reference list from the PDF's text layer, for when GROBID is not available (MCP_PLAN §3.2).

Finds the References heading with the format checker's heading detection, takes the lines up to
the next appendix/checklist heading, and splits them into entries: numbered (`[12]`, `12.`)
when most entries are numbered, else author–year (a new entry starts after a line that ends a
reference and on a line that starts like an author list). Each entry becomes a `Reference` with
only `raw_string`; the resolver derives title, DOI and arXiv id from it.
"""

import re
import unicodedata
from collections import Counter

from reviewer.citations import raw as rawlib
from reviewer.format.layout import find_headings, page_lines
from reviewer.models import Reference

BRACKET = re.compile(r"^\s*\[(\d{1,3})\]\s*")
DOTTED = re.compile(r"^\s*(\d{1,3})\.\s+(?=[A-ZÀ-ɏ])")
_NAME = r"[A-ZÀ-ɏ][\wÀ-ɏ'’`\-]*\.?"
AUTHOR_START = re.compile(
    # 1-4 capitalised name tokens or initials ("Xiang Lisa Li", "A. Vaswani", "Ben Zaken"),
    # then what follows a name in an author list: a comma, "and"/"&", or "et al."
    # (a comma followed by a year is a venue line, "Learning Representations, 2019.", not names)
    rf"^(?:{_NAME}\s+){{0,3}}{_NAME}(?:,(?!\s*\d)|\s+(?:and|&)\s|\s+et\s+al\.)"
)
PAGE_NUMBER = re.compile(r"^\s*\d{1,3}\s*$")
MIN_ENTRIES_FOR_NUMBERED = 3
MAX_ENTRY_CHARS = 1500
"""Longer than any real reference: an entry this long is the last reference with an
undetected appendix glued on. It ends the list and is dropped (the resolver would also spend
minutes matching titles against tens of thousands of characters)."""
MAX_NON_REFERENCE_RUN = 3
"""After this many consecutive entries that do not look like references, the list has ended
(an appendix the heading detection missed)."""


def _clean_lines(lines: list[str], repeated: set[str]) -> list[str]:
    out = []
    for ln in lines:
        s = ln.strip()
        if not s or PAGE_NUMBER.match(s) or s in repeated:
            continue
        out.append(s)
    return out


def _join(parts: list[str]) -> str:
    text = ""
    for p in parts:
        if text.endswith("-") and p[:1].islower():
            text = text[:-1] + p  # de-hyphenate a word broken across lines
        else:
            text = f"{text} {p}" if text else p
    # NFKC: the text layer keeps typographic ligatures ("eﬀicient"), which break title matching
    return " ".join(unicodedata.normalize("NFKC", text).split())


def _split_numbered(lines: list[str], marker: re.Pattern[str]) -> list[str]:
    entries: list[list[str]] = []
    for ln in lines:
        if marker.match(ln):
            entries.append([marker.sub("", ln, count=1)])
        elif entries:
            entries[-1].append(ln)
    return [_join(e) for e in entries]


def _split_author_year(lines: list[str]) -> list[str]:
    entries: list[list[str]] = []
    for ln in lines:
        prev = entries[-1][-1] if entries else ""
        if not entries or (prev.rstrip().endswith(".") and AUTHOR_START.match(ln)):
            entries.append([ln])
        else:
            entries[-1].append(ln)
    return [_join(e) for e in entries]


def reference_lines(pages: list[str]) -> list[str]:
    """Lines of the reference list, with running headers/footers and page numbers removed."""
    headings = find_headings(pages)
    refs = [h for h in headings if h.kind == "references" and h.page >= 2]
    if not refs:
        return []
    start = refs[0]
    stops = [
        h
        for h in headings
        if h.kind in ("appendix", "checklist") and (h.page, h.line) > (start.page, start.line)
    ]
    stop = stops[0] if stops else None
    per_page = [page_lines(p) for p in pages]
    # a line repeated on most pages is a running header or footer
    counts = Counter(ln.strip() for lines in per_page for ln in set(lines) if ln.strip())
    repeated = {ln for ln, n in counts.items() if n >= max(3, len(pages) // 2)}
    lines: list[str] = []
    for page_no in range(start.page, (stop.page if stop else len(pages)) + 1):
        page = per_page[page_no - 1]
        lo = start.line + 1 if page_no == start.page else 0
        hi = stop.line if stop and page_no == stop.page else len(page)
        lines += page[lo:hi]
    return _clean_lines(lines, repeated)


def split_references(lines: list[str]) -> list[str]:
    """Numbered only when the list itself starts with entry 0 or 1 (an appendix's numbered list
    further down must not switch the mode)."""
    for marker in (BRACKET, DOTTED):
        first = marker.match(lines[0]) if lines else None
        if (
            first
            and int(first.group(1)) <= 1
            and sum(bool(marker.match(ln)) for ln in lines) >= MIN_ENTRIES_FOR_NUMBERED
        ):
            return _split_numbered(lines, marker)
    return _split_author_year(lines)


def until_list_ends(entries: list[str]) -> list[str]:
    """Reference-like entries, stopping at the first run of MAX_NON_REFERENCE_RUN entries that
    do not look like references."""
    out: list[str] = []
    run = 0
    for e in entries:
        if len(e) > MAX_ENTRY_CHARS:
            break
        if rawlib.looks_like_reference(e):
            out.append(e)
            run = 0
        else:
            run += 1
            if run >= MAX_NON_REFERENCE_RUN:
                break
    return out


def fallback_references(pages: list[str]) -> list[Reference]:
    raws = until_list_ends(split_references(reference_lines(pages)))
    return [Reference(ref_id=f"f{i}", raw_string=r) for i, r in enumerate(raws)]
