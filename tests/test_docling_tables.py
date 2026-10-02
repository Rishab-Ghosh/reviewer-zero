"""Docling table conversion, caching, and the GROBID-vs-Docling choice. No Docling models run."""

from pathlib import Path
from typing import Any

import pytest
from docling_core.types.doc import (
    BoundingBox,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
    TableCell,
    TableData,
)

from reviewer.config import Settings
from reviewer.models import Table
from reviewer.parse.docling_tables import (
    DoclingError,
    DoclingTableExtractor,
    choose_tables,
    table_label,
    tables_from_document,
    usable_tables,
)


def _cell(text: str, r: int, c: int, rs: int = 1, cs: int = 1) -> TableCell:
    return TableCell(
        text=text,
        row_span=rs,
        col_span=cs,
        start_row_offset_idx=r,
        end_row_offset_idx=r + rs,
        start_col_offset_idx=c,
        end_col_offset_idx=c + cs,
    )


def _add_table(
    doc: DoclingDocument, cells: list[TableCell], n_rows: int, n_cols: int, caption: str, page: int
) -> None:
    prov = ProvenanceItem(page_no=page, bbox=BoundingBox(l=0, t=0, r=1, b=1), charspan=(0, 0))
    cap = doc.add_text(label=DocItemLabel.CAPTION, text=caption) if caption else None
    doc.add_table(
        data=TableData(table_cells=cells, num_rows=n_rows, num_cols=n_cols),
        caption=cap,
        prov=prov,
    )


def _fake_doc() -> DoclingDocument:
    doc = DoclingDocument(name="fake")
    # Table 1: header cell spanning two columns, whitespace to normalize.
    _add_table(
        doc,
        [
            _cell("Model", 0, 0),
            _cell("BLEU  \n scores", 0, 1, cs=2),
            _cell("base", 1, 0),
            _cell("25.8", 1, 1),
            _cell("26.4", 1, 2),
        ],
        2,
        3,
        "Table 1:  Results on\nWMT.",
        6,
    )
    # Degenerate: a single row.
    _add_table(doc, [_cell("a", 0, 0), _cell("b", 0, 1)], 1, 2, "Table 9: junk", 7)
    # No caption; one grid position uncovered by any cell.
    _add_table(doc, [_cell("x", 0, 0), _cell("y", 0, 1), _cell("z", 1, 0)], 2, 2, "", 8)
    # "Tab. 4" caption style.
    _add_table(doc, [_cell(str(i), i // 2, i % 2) for i in range(4)], 2, 2, "Tab. 4 Ablation", 9)
    return doc


def test_tables_from_document() -> None:
    tables, n_degenerate = tables_from_document(_fake_doc())
    assert n_degenerate == 1
    assert [t.label for t in tables] == ["Table 1", "Docling table 3", "Table 4"]
    t1 = tables[0]
    assert t1.caption == "Table 1: Results on WMT."
    assert t1.rows == [["Model", "BLEU scores", "BLEU scores"], ["base", "25.8", "26.4"]]
    assert (t1.source, t1.page, t1.raw_text) == ("docling", 6, None)
    assert tables[1].caption == ""
    assert tables[1].rows == [["x", "y"], ["z", ""]]
    assert tables[2].page == 9


@pytest.mark.parametrize(
    ("caption", "label"),
    [
        ("Table 2: x", "Table 2"),
        ("Table2 FitstoL(N,D)", "Table 2"),
        ("TABLE 3. x", "Table 3"),
        ("Tab. 5 x", "Table 5"),
        ("Table A1: appendix", "Table A1"),
        ("Tables are hard", "Docling table 7"),
        ("", "Docling table 7"),
        ("See Table 2", "Docling table 7"),
    ],
)
def test_table_label(caption: str, label: str) -> None:
    assert table_label(caption, 7) == label


class CountingConverter:
    def __init__(self, doc: Any = None, exc: Exception | None = None) -> None:
        self.doc = doc
        self.exc = exc
        self.calls = 0

    def __call__(self, pdf: Path) -> Any:
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return self.doc


def _pdf(tmp_path: Path, content: bytes = b"%PDF-1.5 fake") -> Path:
    p = tmp_path / "paper.pdf"
    p.write_bytes(content)
    return p


def test_extract_caches_by_content(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    conv = CountingConverter(_fake_doc())
    pdf = _pdf(tmp_path)
    first = DoclingTableExtractor(settings, converter=conv).extract(pdf)
    second = DoclingTableExtractor(settings, converter=conv).extract(pdf)
    assert conv.calls == 1
    assert first == second and len(first) == 3
    # Same bytes under another name: still a hit. Different mode: a miss.
    other = tmp_path / "renamed.pdf"
    other.write_bytes(pdf.read_bytes())
    DoclingTableExtractor(settings, converter=conv).extract(other)
    assert conv.calls == 1
    DoclingTableExtractor(settings, mode="fast", converter=conv).extract(pdf)
    assert conv.calls == 2


def test_extract_failure_raises_and_is_not_cached(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    conv = CountingConverter(exc=RuntimeError("boom"))
    ex = DoclingTableExtractor(settings, converter=conv)
    with pytest.raises(DoclingError, match="boom"):
        ex.extract(_pdf(tmp_path))
    assert list((settings.cache_dir / "docling").iterdir()) == []


def _t(label: str, n_rows: int = 2, raw: str | None = None, source: str = "grobid") -> Table:
    rows = [["a", "b"]] * n_rows if n_rows else None
    return Table(label=label, caption="", rows=rows, raw_text=raw, source=source)  # type: ignore[arg-type]


def test_usable_tables() -> None:
    tables = [_t("ok"), _t("one row", 1), _t("raw", 0, raw="1 2 3"), _t("empty", 0, raw="  ")]
    assert [t.label for t in usable_tables(tables)] == ["ok", "raw"]


def test_choose_tables() -> None:
    g2 = [_t("g1"), _t("g2"), _t("g-bad", 1)]
    d1 = [_t("d1", source="docling")]
    d3 = [_t(f"d{i}", source="docling") for i in range(3)]
    assert choose_tables([], d1) == (d1, "docling")
    assert choose_tables(g2, d3) == (d3, "docling")
    assert choose_tables(g2, d1) == (g2[:2], "grobid")
    assert choose_tables(g2, [_t("d", source="docling"), _t("d", source="docling")])[1] == "grobid"
    assert choose_tables(g2, None) == (g2[:2], "grobid")
    assert choose_tables([_t("bad", 1)], None) == ([], "none")
    assert choose_tables([], []) == ([], "none")


def test_contents_pages_and_author_blocks_are_not_usable():
    from reviewer.models import Table
    from reviewer.parse.docling_tables import usable_tables

    toc = Table(
        label="Docling table 2",
        caption="",
        source="docling",
        rows=[["1 Introduction . . . . . . . 2"], ["2 Background . . . . . . . 3"]],
    )
    authors = Table(
        label="Docling table 1",
        caption="",
        source="docling",
        rows=[["A. Author", "a@x.com"], ["B. Author", "b@y.org"]],
    )
    real = Table(
        label="Table 1", caption="c", source="docling", rows=[["Model", "BLEU"], ["X", "28.4"]]
    )
    assert usable_tables([toc, authors, real]) == [real]
