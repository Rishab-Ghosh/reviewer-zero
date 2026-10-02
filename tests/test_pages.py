"""Page attribution (tei.page_texts) and page coverage (parse.pages); no network."""

import subprocess
from pathlib import Path

import pytest

from reviewer.models import PageCoverage
from reviewer.parse import pages as pages_mod
from reviewer.parse.pages import (
    PdfToTextError,
    count_chars,
    count_words,
    fallback_sections,
    page_coverage,
    page_paragraphs,
    pdftotext_pages,
    split_pages,
    token_presence,
)
from reviewer.parse.tei import (
    UNATTRIBUTED_PAGE,
    page_texts,
    parse_coords,
    parse_tei,
    split_by_boxes,
)
from tests.fulltext import fulltext_tei

FIXTURES = Path(__file__).parent / "fixtures"

_NS = 'xmlns="http://www.tei-c.org/ns/1.0"'


def _tei(header: str = "", body: str = "", back: str = "") -> str:
    return (
        f"<TEI {_NS}><teiHeader>{header}</teiHeader>"
        f"<text><body>{body}</body><back>{back}</back></text></TEI>"
    )


# --------------------------------------------------------------------------- tei.page_texts


def test_parse_coords() -> None:
    assert parse_coords("1,2.0,3,4,5;2,1,1,1,1;2,0,0,1,1") == [1, 2, 2]
    assert parse_coords(None) == []
    assert parse_coords("garbage;;x,1") == []


def test_split_by_boxes_proportional_and_word_safe() -> None:
    text = "aaaa bbbb cccc dddd"
    assert split_by_boxes(text, [3, 3]) == [(3, text)]
    pieces = split_by_boxes(text, [1, 2, 2, 2])  # 1/4 to page 1, 3/4 to page 2
    assert [p for p, _ in pieces] == [1, 2]
    assert pieces[0][1] == "aaaa"
    assert " ".join(t for _, t in pieces) == text
    assert split_by_boxes(text, []) == []


def test_page_texts_no_nested_double_counting() -> None:
    xml = _tei(
        body=(
            '<div><head coords="1,0,0,1,1">Intro</head>'
            '<p coords="1,0,0,1,1;2,0,0,1,1">'
            '<s coords="1,0,0,1,1">Alpha <ref coords="1,0,0,1,1">[1]</ref> beta.</s>'
            '<s coords="2,0,0,1,1">Gamma delta.</s></p>'
            '<p coords="2,0,0,1,1">Unsegmented para.</p>'
            '<formula coords="2,0,0,1,1">x=y<label>(1)</label></formula></div>'
        ),
        back=(
            '<div type="references"><listBibl><biblStruct coords="3,0,0,1,1">'
            '<analytic><title coords="3,0,0,1,1">Some Title</title></analytic>'
            '<note type="raw_reference">Doe. Some Title. 2020.</note></biblStruct></listBibl></div>'
        ),
    )
    pages = page_texts(xml)
    assert pages[1] == "Intro\nAlpha [1] beta."
    assert pages[2] == "Gamma delta.\nUnsegmented para.\nx=y (1)"
    assert pages[3] == "Doe. Some Title. 2020."


def test_page_texts_header_fallback_and_unattributed() -> None:
    header = (
        "<fileDesc><titleStmt><title type='main'>My Paper</title></titleStmt>"
        "<sourceDesc><biblStruct><analytic><author><persName>"
        "<forename>Ann</forename><surname>Lee</surname></persName></author>"
        "</analytic></biblStruct></sourceDesc></fileDesc>"
        '<profileDesc><abstract><div><p><s coords="2,0,0,1,1">Abs.</s></p></div></abstract>'
        "</profileDesc>"
    )
    body = "<div><p><s>No coords here.</s></p></div>"
    pages = page_texts(_tei(header=header, body=body))
    assert pages[1] == "My Paper\nAnn Lee"
    assert pages[2] == "Abs."
    assert pages[UNATTRIBUTED_PAGE] == "No coords here."


def test_page_texts_bad_xml() -> None:
    assert page_texts("") == {}


def test_attention_page_texts_real_fixture() -> None:
    pages = page_texts(fulltext_tei("1706.03762"))
    assert sorted(pages) == list(range(1, 16))  # 15-page PDF, nothing unattributed
    assert "Attention Is All You Need" in pages[1]
    assert "The dominant sequence transduction models" in pages[1]
    assert "Scaled Dot-Product Attention" in pages[4]

    # No double counting: total attributed text ≈ the text parse_tei keeps (the only extra is
    # the three footnotes, which parse_tei drops).
    paper = parse_tei(fulltext_tei("1706.03762"), paper_id="1706.03762", source_path="x.pdf")
    c = count_chars
    expected = c(paper.title) + c(paper.abstract)
    expected += sum(c(a.name) + c(a.affiliation or "") for a in paper.authors)
    expected += sum(
        c(s.heading) + c(s.number or "") + sum(c(p) for p in s.paragraphs) for s in paper.sections
    )
    expected += sum(c(f.label) + c(f.caption) for f in paper.figures)
    expected += sum(
        c(t.label) + c(t.caption) + sum(c(x) for row in t.rows or [] for x in row)
        for t in paper.tables
    )
    expected += sum(c(r.raw_string) for r in paper.references)
    got = sum(c(t) for t in pages.values())
    assert 0.97 < got / expected < 1.05


def test_parse_tei_does_not_leak_coords() -> None:
    paper = parse_tei(fulltext_tei("1706.03762"), paper_id="1706.03762", source_path="x.pdf")
    dumped = paper.model_dump_json()
    assert "coords" not in dumped
    assert "1,143.56" not in dumped


# --------------------------------------------------------------------------- counting / pdftotext


def test_counts() -> None:
    assert count_chars(" a b\n\tc ") == 3
    assert count_words(" a  bb\nccc ") == 3
    assert count_chars("") == count_words("") == 0


def test_split_pages_drops_trailing_chunk() -> None:
    assert split_pages("one\fTwo\f") == ["one", "Two"]
    assert split_pages("one\f\ftwo\f") == ["one", "", "two"]  # empty middle page kept


def test_pdftotext_pages_runs_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="p1 é\fp2\f".encode(), stderr=b"")

    monkeypatch.setattr(pages_mod.shutil, "which", lambda _: "/bin/pdftotext")
    monkeypatch.setattr(pages_mod.subprocess, "run", fake_run)
    assert pdftotext_pages(tmp_path / "a.pdf") == ["p1 é", "p2"]
    assert calls == [["/bin/pdftotext", "-enc", "UTF-8", str(tmp_path / "a.pdf"), "-"]]


def test_pdftotext_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pages_mod.shutil, "which", lambda _: None)
    with pytest.raises(PdfToTextError, match="pdftotext not found"):
        pdftotext_pages(tmp_path / "a.pdf")


def test_pdftotext_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pages_mod.shutil, "which", lambda _: "/bin/pdftotext")
    monkeypatch.setattr(
        pages_mod.subprocess,
        "run",
        lambda cmd, **_: subprocess.CompletedProcess(cmd, 1, stdout=b"", stderr=b"bad pdf"),
    )
    with pytest.raises(PdfToTextError, match="bad pdf"):
        pdftotext_pages(tmp_path / "a.pdf")


# --------------------------------------------------------------------------- coverage


def test_page_coverage_math() -> None:
    pdf = ["abcd efgh", "", "abcd", "ab"]
    parse = {0: "ignored", 1: "abcd efgh", 2: "stray", 3: "a", 4: "abcdef", 9: "beyond"}
    cov = page_coverage(parse, pdf)
    assert [c.page for c in cov] == [1, 2, 3, 4]
    assert cov[0].coverage == 1.0 and not cov[0].flagged
    assert cov[0].parse_words == cov[0].pdftotext_words == 2
    assert cov[1].coverage is None and not cov[1].flagged  # empty text layer
    assert cov[2].coverage == 0.25 and cov[2].flagged
    assert cov[3].coverage == 3.0 and not cov[3].flagged  # > 1 kept raw
    assert page_coverage({}, ["x"])[0].flagged
    assert not page_coverage({1: "ab"}, ["abcd"], threshold=0.5)[0].flagged  # 0.5 not < 0.5


def test_page_paragraphs() -> None:
    text = "First line\nsecond   line\n\n  \n\nNext\tpara\n\n\n"
    assert page_paragraphs(text) == ["First line second line", "Next para"]


def test_fallback_sections() -> None:
    pdf = ["full page", "Low one\ncontinues\n\nSecond para", ""]
    cov = page_coverage({1: "full page", 2: "Low"}, pdf)
    sections = fallback_sections(cov, pdf)
    assert len(sections) == 1
    s = sections[0]
    assert s.heading == "Page 2 (pdftotext fallback)"
    assert s.level == 1 and s.fallback and s.fallback_page == 2
    assert s.paragraphs == ["Low one continues", "Second para"]
    flagged_out_of_range = PageCoverage(
        page=7,
        parse_chars=0,
        parse_words=0,
        pdftotext_chars=5,
        pdftotext_words=1,
        coverage=0.0,
        flagged=True,
    )
    assert fallback_sections([flagged_out_of_range], pdf) == []


def test_token_presence_case_and_boundaries() -> None:
    pdf = ["We use DiD and REL (REL-2) with Winkler.", "RELU did res"]
    parse = {1: "We use did and RELU with Winkler's", 2: "REL DiD"}
    out = token_presence(parse, pdf, ["REL", "RES", "Winkler", "DiD"], [1, 2, 5])
    assert out[1] == {
        "REL": (True, False),
        "RES": (False, False),
        "Winkler": (True, True),
        "DiD": (True, False),
    }
    assert out[2] == {
        "REL": (False, True),
        "RES": (False, False),
        "Winkler": (False, False),
        "DiD": (False, True),
    }
    assert out[5] == {t: (False, False) for t in ["REL", "RES", "Winkler", "DiD"]}


def test_with_abstract_and_with_tables_rules():
    from reviewer.models import ParsedPaper, Table
    from reviewer.parse.commands import with_abstract, with_tables

    paper = ParsedPaper(
        paper_id="1706.03762", source_path="x.pdf", title="T", abstract="grobid abs"
    )
    a = with_abstract(paper, "arxiv abs")
    assert (a.abstract, a.abstract_source, a.grobid_abstract) == (
        "arxiv abs",
        "arxiv",
        "grobid abs",
    )
    b = with_abstract(paper, None)
    assert b.abstract_source == "grobid" and any(
        "unavailable" in w for w in b.parse_report.warnings
    )
    own = paper.model_copy(update={"paper_id": "own-x-12345678"})
    assert with_abstract(own, None).parse_report.warnings == []

    cap_only = Table(label="Table 1", caption="c")
    g = Table(label="Table 2", caption="c", rows=[["a", "b"], ["1", "2"]])
    d1 = Table(label="Table 1", caption="c", rows=[["a", "b"], ["1", "2"]], source="docling")
    d2 = Table(label="Table 2", caption="c", rows=[["a", "b"], ["3", "4"]], source="docling")
    p2 = paper.model_copy(update={"tables": [cap_only, g]})
    kept = with_tables(p2, [d1])  # tie on usable count -> GROBID, caption-only table kept
    assert kept.table_source == "grobid" and kept.tables == [cap_only, g]
    won = with_tables(p2, [d1, d2])
    assert won.table_source == "docling" and won.tables == [d1, d2]
    assert with_tables(paper, None).table_source == "none"
