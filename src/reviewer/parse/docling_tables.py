"""Docling as a table-only extractor, plus the rule for choosing GROBID's or Docling's tables.

Docling's layout model + TableFormer recover cell structure for tables GROBID misses or
flattens. We use nothing else from Docling: text, sections and references stay GROBID's.

Cell grid convention: `rows` is Docling's `TableData.grid`, so a cell spanning several rows or
columns has its text *repeated* in every grid position it covers (e.g. a header spanning three
columns appears three times). Positions no cell covers are empty strings. Whitespace inside a
cell is collapsed to single spaces.
"""

import logging
import re
import time
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

from reviewer.cache import DiskCache, content_key, sha256_file
from reviewer.config import Settings
from reviewer.models import Table

logger = logging.getLogger(__name__)

TableMode = Literal["accurate", "fast"]
TableSource = Literal["grobid", "docling", "none"]

CACHE_SCHEMA = "docling-tables-v1"
MIN_ROWS = 2
MIN_COLS = 2

_LABEL = re.compile(r"^\s*(?:Table|Tab\.?)\s*([A-Z]?\d+[a-z]?)\b", re.I)

# A converter takes a PDF path and returns a DoclingDocument (or anything with the same
# `.tables` / `caption_text(doc)` / `.data.grid` / `.prov` shape).
Converter = Callable[[Path], Any]


class DoclingError(Exception):
    """Docling failed to convert a PDF. The caller records the failure; nothing is cached."""


def docling_version() -> str:
    return version("docling")


def canonical_options(mode: TableMode) -> str:
    """Every option that changes Docling's table output, as one stable string (a cache key part)."""
    return f"{CACHE_SCHEMA};ocr=off;table_structure=on;tableformer={mode};cell_matching=on"


def docling_cache_key(pdf_sha256: str, mode: TableMode, docling_ver: str | None = None) -> str:
    return content_key(pdf_sha256, docling_ver or docling_version(), canonical_options(mode))


def _norm(text: str) -> str:
    return " ".join(text.split())


def table_label(caption: str, k: int) -> str:
    """'Table N' when the caption starts with Table/Tab. N, else 'Docling table k' (1-based)."""
    m = _LABEL.match(caption)
    return f"Table {m.group(1)}" if m else f"Docling table {k}"


def tables_from_document(doc: Any) -> tuple[list[Table], int]:
    """Convert a DoclingDocument's tables to our Table models.

    Returns (tables, n_degenerate): tables with fewer than MIN_ROWS rows or MIN_COLS columns
    are dropped and only counted. `k` in fallback labels counts every Docling table in document
    order (degenerate ones included), so labels are stable across filter changes.
    """
    out: list[Table] = []
    n_degenerate = 0
    for k, item in enumerate(doc.tables, start=1):
        grid = item.data.grid
        rows = [[_norm(cell.text or "") for cell in row] for row in grid]
        n_cols = max((len(r) for r in rows), default=0)
        if len(rows) < MIN_ROWS or n_cols < MIN_COLS:
            n_degenerate += 1
            continue
        caption = _norm(item.caption_text(doc) or "")
        page = item.prov[0].page_no if item.prov else None
        out.append(
            Table(
                label=table_label(caption, k),
                caption=caption,
                rows=rows,
                raw_text=None,
                source="docling",
                page=page,
            )
        )
    return out, n_degenerate


def _default_converter(mode: TableMode) -> Converter:
    """Build a Docling DocumentConverter: OCR off, TableFormer on. Imported lazily (slow)."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.do_ocr = False
    opts.do_table_structure = True
    opts.table_structure_options.mode = (
        TableFormerMode.ACCURATE if mode == "accurate" else TableFormerMode.FAST
    )
    opts.table_structure_options.do_cell_matching = True
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )

    def convert(pdf: Path) -> Any:
        return converter.convert(pdf).document

    return convert


class DoclingTableExtractor:
    """Runs Docling on a PDF and returns its tables, cached by (PDF sha256, version, options)."""

    def __init__(
        self,
        settings: Settings,
        mode: TableMode = "accurate",
        converter: Converter | None = None,
    ) -> None:
        self.settings = settings
        self.mode = mode
        self._converter = converter
        self.cache = DiskCache(settings.paper_cache, "docling", ".json")

    def _convert(self, pdf: Path) -> Any:
        if self._converter is None:
            self._converter = _default_converter(self.mode)
        return self._converter(pdf)

    def extract(self, pdf: Path) -> list[Table]:
        """Docling tables for `pdf`. Raises DoclingError if Docling fails (not cached)."""
        key = docling_cache_key(sha256_file(pdf), self.mode)
        cached = self.cache.get_json(key)
        if cached is not None:
            return [Table.model_validate(t) for t in cached["tables"]]

        start = time.monotonic()
        try:
            doc = self._convert(pdf)
            tables, n_degenerate = tables_from_document(doc)
        except Exception as e:  # Docling raises many exception types; the caller records it.
            raise DoclingError(f"docling failed on {pdf.name}: {type(e).__name__}: {e}") from e
        elapsed = time.monotonic() - start
        logger.info(
            "docling %s: %d tables kept, %d degenerate skipped (%.1fs, tableformer=%s)",
            pdf.name,
            len(tables),
            n_degenerate,
            elapsed,
            self.mode,
        )
        self.cache.put_json(
            key,
            {
                "docling_version": docling_version(),
                "options": canonical_options(self.mode),
                "n_degenerate": n_degenerate,
                "seconds": round(elapsed, 1),
                "tables": [t.model_dump() for t in tables],
            },
        )
        return tables


# --------------------------------------------------------------------------- choosing


_LEADER_RE = re.compile(r"(?:\.\s?){4,}")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


def looks_like_non_table(table: Table) -> bool:
    """Layout detectors also box contents pages, lists of figures/tables (dot leaders) and author
    blocks (emails). Treat a table as one of those when at least half its rows have a dot
    leader, or at least two rows contain an email address."""
    rows = [" ".join(r) for r in (table.rows or [])]
    if not rows:
        return False
    leaders = sum(bool(_LEADER_RE.search(r)) for r in rows)
    emails = sum(bool(_EMAIL_RE.search(r)) for r in rows)
    return leaders * 2 >= len(rows) or emails >= 2


def usable_tables(tables: list[Table]) -> list[Table]:
    """Tables with real content: at least 2 cell rows, or non-empty flattened text; contents
    pages and author blocks mistaken for tables are excluded."""
    return [
        t
        for t in tables
        if ((t.rows and len(t.rows) >= 2) or (t.raw_text or "").strip())
        and not looks_like_non_table(t)
    ]


def choose_tables(
    grobid: list[Table], docling: list[Table] | None
) -> tuple[list[Table], TableSource]:
    """Pick one parser's tables for a paper (never a mix).

    Docling wins when it has more usable tables than GROBID (which covers GROBID having none).
    Ties and a missing/failed Docling run (`docling is None`) keep GROBID. "none" when neither
    has a usable table. Only usable tables are returned.
    """
    g = usable_tables(grobid)
    d = usable_tables(docling or [])
    if not g and not d:
        return [], "none"
    if len(d) > len(g):
        return d, "docling"
    return g, "grobid"
