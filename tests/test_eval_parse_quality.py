from reviewer.eval.parse_quality import (
    lost_tables,
    parse_quality_row,
    referenced_table_keys,
    render_parse_quality,
    suspicious_authors,
)
from reviewer.models import Author, Figure, Section, Table

from .test_eval_helpers import parsed


def _paper(paragraphs, tables=(), figures=(), authors=()):
    p = parsed("2501.00001")
    p.sections = [Section(heading="Results", paragraphs=list(paragraphs))]
    p.tables = list(tables)
    p.figures = list(figures)
    p.authors = list(authors)
    return p


def test_referenced_table_keys():
    keys = referenced_table_keys(
        ["See Table 3 and Tab. 4.", "Tables 5 and 6 show", "cf. Table A1", "a stable 9 result"]
    )
    assert keys == {"3", "4", "5", "6", "A1"}


def test_lost_tables_by_number():
    p = _paper(
        ["As Table 1 shows, and Table 2 confirms (Tab. 3)."],
        tables=[
            Table(label="Table 2", caption="Results", rows=[["a", "b"]]),
            Table(label="3", caption="Ablation", raw_text="x y"),
        ],
        figures=[Figure(label="Figure 1", caption="Compared with Table 4.")],
    )
    assert lost_tables(p) == ["Table 1", "Table 4"]


def test_lost_tables_ignores_appendix_unless_same_scheme():
    p = _paper(["Table 1 and Table A1."], tables=[Table(label="Table 1", caption="", rows=[["x"]])])
    assert lost_tables(p) == []
    p.tables.append(Table(label="Table A2", caption="", rows=[["y"]]))
    assert lost_tables(p) == ["Table A1"]


def test_empty_parsed_table_reported():
    p = _paper(["Table 1."], tables=[Table(label="Table 1", caption="c", rows=[], raw_text=" ")])
    assert lost_tables(p) == ["Table 1 (parsed, no rows or text)"]


def test_label_from_caption_when_label_is_xml_id():
    p = _paper(["Table 2."], tables=[Table(label="tab_0", caption="Table 2 results", rows=[["x"]])])
    assert lost_tables(p) == []


def test_suspicious_authors():
    authors = [
        Author(name="Ashish Vaswani"),
        Author(name="Google Brain"),
        Author(name="Google Research"),
        Author(name="Noam"),
        Author(name="jdoe@example.com"),
        Author(name="Stanford University"),
    ]
    out = suspicious_authors(_paper([], authors=authors))
    assert len(out) == 5
    assert out[0] == "Google Brain (organization word 'Google')"
    assert "Noam (single token)" in out
    assert any(s.startswith("jdoe@example.com (") and "digit or @" in s for s in out)
    assert not any("Vaswani" in s for s in out)


def test_affiliation_smear_heuristic():
    same = [Author(name=f"Person Number{c}", affiliation="Google Brain") for c in "ABCD"]
    out = suspicious_authors(_paper([], authors=same))
    assert out == [
        "heuristic: all 4 authors have the identical affiliation 'Google Brain' "
        "(possible affiliation smear)"
    ]
    assert suspicious_authors(_paper([], authors=same[:3])) == []
    mixed = [*same[:3], Author(name="Other Person", affiliation="MIT")]
    assert suspicious_authors(_paper([], authors=mixed)) == []


def test_render_parse_quality():
    p = _paper(["Table 1."], authors=[Author(name="Google Brain")])
    text = render_parse_quality([parse_quality_row(p, "famous")])
    header, _rule, row = text.splitlines()
    assert header.split() == [
        "paper_id",
        "tier",
        "tables",
        "lost",
        "tables",
        "suspicious",
        "authors",
    ]
    assert row.startswith("2501.00001") and "famous" in row and "Table 1" in row
    assert "Google Brain" in row
    assert "(no parsed papers)" in render_parse_quality([])
