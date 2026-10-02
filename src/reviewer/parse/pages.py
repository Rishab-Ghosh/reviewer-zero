"""Per-page coverage of the GROBID parse against the PDF's text layer (pdftotext).

The parse's page texts come from `tei.page_texts` (GROBID coordinates); the reference is
`pdftotext`, one chunk per page. Pages where the parse recovered much less text than the text
layer holds are flagged, and their pdftotext text can be spliced in as fallback sections.
"""

import logging
import re
import shutil
import subprocess
from pathlib import Path

from reviewer.models import PageCoverage, Section

logger = logging.getLogger(__name__)

PDFTOTEXT = "pdftotext"
DEFAULT_THRESHOLD = 0.5
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


class PdfToTextError(RuntimeError):
    """pdftotext is missing or failed."""


def pdftotext_pages(pdf: Path, timeout_s: float = 120.0) -> list[str]:
    """Text layer of `pdf`, one string per page (index i = page i+1).

    Runs `pdftotext -enc UTF-8 <pdf> -` once and splits on form feeds (pdftotext ends every page
    with one, so the trailing empty chunk is dropped).
    """
    exe = shutil.which(PDFTOTEXT)
    if exe is None:
        raise PdfToTextError(
            "pdftotext not found on PATH; install poppler (e.g. `brew install poppler`)"
        )
    try:
        proc = subprocess.run(
            [exe, "-enc", "UTF-8", str(pdf), "-"],
            capture_output=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise PdfToTextError(f"pdftotext timed out after {timeout_s:.0f}s on {pdf.name}") from exc
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()[:200]
        raise PdfToTextError(f"pdftotext failed on {pdf.name} (exit {proc.returncode}): {err}")
    return split_pages(proc.stdout.decode("utf-8", errors="replace"))


def split_pages(text: str) -> list[str]:
    """Split pdftotext output on form feeds, dropping the empty chunk after the last one."""
    chunks = text.split("\f")
    if chunks and not chunks[-1].strip():
        chunks = chunks[:-1]
    return chunks


def count_chars(text: str) -> int:
    """Non-whitespace characters."""
    return sum(1 for c in text if not c.isspace())


def count_words(text: str) -> int:
    """Whitespace-separated tokens."""
    return len(text.split())


def page_coverage(
    parse_pages: dict[int, str], pdf_pages: list[str], threshold: float = DEFAULT_THRESHOLD
) -> list[PageCoverage]:
    """One entry per PDF page 1..N comparing parse text to the text layer.

    coverage = parse_chars / pdftotext_chars, kept raw (it can exceed 1 when GROBID adds or
    duplicates text); None when the page has no text layer, and such pages are never flagged.
    flagged = coverage < threshold. Parse text on page 0 (unattributed) or beyond N is ignored.
    """
    out: list[PageCoverage] = []
    for page, pdf_text in enumerate(pdf_pages, start=1):
        parse_text = parse_pages.get(page, "")
        parse_chars = count_chars(parse_text)
        pdf_chars = count_chars(pdf_text)
        coverage = parse_chars / pdf_chars if pdf_chars else None
        out.append(
            PageCoverage(
                page=page,
                parse_chars=parse_chars,
                parse_words=count_words(parse_text),
                pdftotext_chars=pdf_chars,
                pdftotext_words=count_words(pdf_text),
                coverage=coverage,
                flagged=coverage is not None and coverage < threshold,
            )
        )
    extra = sorted(p for p in parse_pages if p > len(pdf_pages))
    if extra:
        logger.warning(
            "parse text attributed to pages beyond the PDF's %d: %s", len(pdf_pages), extra
        )
    return out


def page_paragraphs(text: str) -> list[str]:
    """Blank-line-separated paragraphs, line breaks joined by spaces, whitespace collapsed."""
    paragraphs = (" ".join(chunk.split()) for chunk in _PARAGRAPH_BREAK.split(text))
    return [p for p in paragraphs if p]


def fallback_sections(coverage: list[PageCoverage], pdf_pages: list[str]) -> list[Section]:
    """A level-1 fallback Section holding the pdftotext text of each flagged page."""
    sections: list[Section] = []
    for entry in coverage:
        if not entry.flagged or not 1 <= entry.page <= len(pdf_pages):
            continue
        sections.append(
            Section(
                heading=f"Page {entry.page} (pdftotext fallback)",
                level=1,
                paragraphs=page_paragraphs(pdf_pages[entry.page - 1]),
                fallback=True,
                fallback_page=entry.page,
            )
        )
    return sections


def token_pattern(token: str) -> re.Pattern[str]:
    """Case-sensitive match of `token` not glued to other letters/digits/underscores.

    'DiD' does not match 'did'; 'REL' does not match inside 'RELU'; 'REL' matches 'REL-2' or
    '(REL)'.
    """
    return re.compile(rf"(?<![\w]){re.escape(token)}(?![\w])")


def token_presence(
    parse_pages: dict[int, str], pdf_pages: list[str], tokens: list[str], pages: list[int]
) -> dict[int, dict[str, tuple[bool, bool]]]:
    """For each page, token → (present in pdftotext page text, present in parse page text).

    Used to check whether inline tokens (e.g. math-ish acronyms) survive the parse. Pages outside
    1..N have empty pdftotext text.
    """
    patterns = {t: token_pattern(t) for t in tokens}
    out: dict[int, dict[str, tuple[bool, bool]]] = {}
    for page in pages:
        pdf_text = pdf_pages[page - 1] if 1 <= page <= len(pdf_pages) else ""
        parse_text = parse_pages.get(page, "")
        out[page] = {
            t: (bool(pat.search(pdf_text)), bool(pat.search(parse_text)))
            for t, pat in patterns.items()
        }
    return out
