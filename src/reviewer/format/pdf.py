"""PDF text layer and document metadata: poppler (pdftotext / pdfinfo) when installed, else
pdfplumber + pypdf (MCP_PLAN §3.3: the uvx package cannot install poppler)."""

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from reviewer.parse.pages import PdfToTextError, pdftotext_pages

METADATA_FIELDS = ("Title", "Author", "Subject", "Keywords", "Creator", "Producer")


TextEngine = Literal["pdftotext", "pdfplumber", "pypdf"]


@dataclass(frozen=True)
class PdfDoc:
    pages: list[str]
    metadata: dict[str, str]
    engine: TextEngine = "pdftotext"


def pdf_metadata(pdf: Path, timeout_s: float = 60.0) -> dict[str, str]:
    """Document-info fields from `pdfinfo` (empty values dropped)."""
    exe = shutil.which("pdfinfo")
    if exe is None:
        raise PdfToTextError("pdfinfo not found on PATH; install poppler")
    proc = subprocess.run(
        [exe, "-enc", "UTF-8", str(pdf)], capture_output=True, check=False, timeout=timeout_s
    )
    out: dict[str, str] = {}
    for line in proc.stdout.decode("utf-8", errors="replace").splitlines():
        key, sep, value = line.partition(":")
        if sep and key in METADATA_FIELDS and value.strip():
            out[key] = value.strip()
    return out


def page_count(pdf: Path, timeout_s: float = 60.0) -> int | None:
    exe = shutil.which("pdfinfo")
    if exe is None:
        return None
    proc = subprocess.run([exe, str(pdf)], capture_output=True, check=False, timeout=timeout_s)
    for line in proc.stdout.decode("utf-8", errors="replace").splitlines():
        if line.startswith("Pages:"):
            return int(line.split()[1])
    return None


def pages_one_by_one(pdf: Path, n: int, timeout_s: float = 60.0) -> list[str]:
    """pdftotext per page. Used when the text layer contains extra form feeds, so splitting the
    whole-document output on them would misalign pages (seen on 2312.00752: 63 chunks for 36
    pages)."""
    exe = shutil.which("pdftotext")
    if exe is None:
        raise PdfToTextError("pdftotext not found on PATH; install poppler")
    out = []
    for i in range(1, n + 1):
        proc = subprocess.run(
            [exe, "-enc", "UTF-8", "-f", str(i), "-l", str(i), str(pdf), "-"],
            capture_output=True,
            check=False,
            timeout=timeout_s,
        )
        out.append(proc.stdout.decode("utf-8", errors="replace").replace("\f", "\n"))
    return out


def poppler_available() -> bool:
    return shutil.which("pdftotext") is not None and shutil.which("pdfinfo") is not None


def pdfplumber_pages(pdf: Path) -> list[str]:
    """One string per page (reading order, lines joined by newlines), like pdftotext's
    default mode. `x_tolerance=1`: pdfplumber's default (3) runs tightly kerned words
    together ("Underreviewasaconference..."), which breaks every header and heading regex."""
    import pdfplumber

    with pdfplumber.open(pdf) as doc:
        return [page.extract_text(x_tolerance=1) or "" for page in doc.pages]


def pypdf_pages(pdf: Path) -> list[str]:
    """One string per page in content-stream order (LaTeX writes two-column pages column by
    column, so this keeps columns apart where pdfplumber's line order mixes them)."""
    import logging

    from pypdf import PdfReader

    logging.getLogger("pypdf").setLevel(logging.ERROR)  # font-encoding notices, not errors
    return [page.extract_text() or "" for page in PdfReader(pdf).pages]


def pypdf_metadata(pdf: Path) -> dict[str, str]:
    """The same document-info fields as `pdf_metadata`, read with pypdf."""
    from pypdf import PdfReader

    info = PdfReader(pdf).metadata or {}
    out: dict[str, str] = {}
    for field in METADATA_FIELDS:
        value = info.get(f"/{field}")
        if value is not None and str(value).strip():
            out[field] = str(value).strip()
    return out


def load_pdf(pdf: Path, engine: TextEngine | None = None) -> PdfDoc:
    """Pages and metadata. `engine=None`: pdftotext when poppler is installed, else pypdf (the
    closer of the two pure-Python fallbacks on the 30 public dev papers; DECISIONS M5-1)."""
    if engine is None:
        engine = "pdftotext" if poppler_available() else "pypdf"
    if engine == "pdfplumber":
        return PdfDoc(pages=pdfplumber_pages(pdf), metadata=pypdf_metadata(pdf), engine=engine)
    if engine == "pypdf":
        return PdfDoc(pages=pypdf_pages(pdf), metadata=pypdf_metadata(pdf), engine=engine)
    pages = pdftotext_pages(pdf)
    n = page_count(pdf)
    if n is not None and n != len(pages):
        pages = pages_one_by_one(pdf, n)
    return PdfDoc(pages=pages, metadata=pdf_metadata(pdf), engine=engine)
