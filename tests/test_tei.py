"""TEI parser tests against real GROBID 0.9.1 (CRF) output checked in under tests/fixtures/."""

from pathlib import Path

import pytest

from reviewer.models import ParsedPaper
from reviewer.parse.contributions import find_contributions
from reviewer.parse.ids import paper_id_for, slugify
from reviewer.parse.tei import level_from_number, parse_tei, split_heading_number
from tests.fulltext import fulltext_tei

FIXTURES = Path(__file__).parent / "fixtures"


def _load(paper_id: str) -> ParsedPaper:
    xml = fulltext_tei(paper_id)
    return parse_tei(xml, paper_id=paper_id, source_path=f"data/papers/arxiv/{paper_id}.pdf")


@pytest.fixture(scope="module")
def attention() -> ParsedPaper:
    return _load("1706.03762")


@pytest.fixture(scope="module")
def stepwise() -> ParsedPaper:
    return _load("2511.07364")


# --------------------------------------------------------------------------- Attention


def test_attention_header(attention: ParsedPaper) -> None:
    assert attention.title == "Attention Is All You Need"
    assert attention.abstract.startswith("The dominant sequence transduction models")
    assert "BLEU" in attention.abstract
    names = [a.name for a in attention.authors]
    assert "Ashish Vaswani" in names
    assert "Łukasz Kaiser" in names


def test_attention_sections(attention: ParsedPaper) -> None:
    # 22 numbered sections (7 top-level + 15 subsections, matching the paper) plus 4 unnumbered
    # back-matter/appendix divs GROBID emits: 'Attention Visualizations', two junk
    # 'Input-Input Layer5' divs made from the appendix attention plots, and 'Acknowledgements'.
    assert attention.parse_report.n_sections == 26
    numbered = [s for s in attention.sections if s.number]
    assert len(numbered) == 22
    assert [s.heading for s in attention.sections[:3]] == [
        "Introduction",
        "Background",
        "Model Architecture",
    ]
    by_number = {s.number: s for s in attention.sections if s.number}
    assert by_number["3.2.1"].heading == "Scaled Dot-Product Attention"
    assert by_number["3.2.1"].level == 3
    assert by_number["3.2"].level == 2
    assert by_number["7"].heading == "Conclusion"
    assert attention.sections[-1].heading == "Acknowledgements"


def test_attention_formula_folded_into_paragraph(attention: ParsedPaper) -> None:
    sec = next(s for s in attention.sections if s.number == "3.2.1")
    joined = " ".join(sec.paragraphs)
    assert "Attention(Q, K, V ) = softmax( QK T √ d k )V (1)" in joined
    # formula text is appended to the paragraph that introduces it, not a stray paragraph
    assert not any(p.startswith("√") for p in sec.paragraphs)


def test_attention_footnote_markers_removed(attention: ParsedPaper) -> None:
    text = " ".join(p for s in attention.sections for p in s.paragraphs)
    assert "foot_0" not in text


def test_attention_references(attention: ParsedPaper) -> None:
    # The paper's bibliography has 40 entries.
    assert attention.parse_report.n_references == 40
    refs = {r.ref_id: r for r in attention.references}
    ba = refs["b0"]
    assert ba.arxiv_id == "1607.06450"
    assert ba.year == 2016
    assert "Layer normalization" in ba.raw_string
    bahdanau = refs["b1"]
    assert bahdanau.title == "Neural machine translation by jointly learning to align and translate"
    assert bahdanau.arxiv_id == "1409.0473"  # from the untyped 'CoRR, abs/1409.0473' idno
    assert any(r.doi for r in attention.references)


def test_attention_citations_resolve(attention: ParsedPaper) -> None:
    ref_ids = {r.ref_id for r in attention.references}
    report = attention.parse_report
    assert report.n_in_text_citations == len(attention.in_text_citations) > 0
    assert report.n_resolved_citations >= 1
    assert report.n_unresolved_citations == 0
    first = attention.in_text_citations[0]
    assert first.section_heading == "Introduction"
    assert first.sentence.startswith("Recurrent neural networks, long short-term memory")
    assert first.ref_ids[0] == "b12"
    assert all(rid in ref_ids for rid in first.ref_ids)


def test_attention_figures_and_tables(attention: ParsedPaper) -> None:
    # GROBID recovers Figures 1-2 only (3-5 are swallowed into appendix text) and Tables 2-4
    # only (Table 1 is lost entirely).
    assert [f.label for f in attention.figures] == ["Figure 1", "Figure 2"]
    assert attention.figures[0].caption == "The Transformer -model architecture."
    assert [t.label for t in attention.tables] == ["Table 2", "Table 3", "Table 4"]
    parsing = attention.tables[2]
    assert parsing.rows is not None
    assert parsing.rows[0] == ["Parser", "Training", "WSJ 23 F1"]
    assert ["Transformer (4 layers)", "semi-supervised", "92.7"] in parsing.rows
    bleu = attention.tables[0]
    assert bleu.rows is not None
    assert any(row[0] == "Transformer (big)" and "28.4" in row for row in bleu.rows)


def test_attention_contributions_truthfully_absent(attention: ParsedPaper) -> None:
    # The Transformer paper has no contributions statement: its introduction ends with
    # "In this work we propose the Transformer, ..." and never says "our contributions",
    # "we make the following contributions", "in summary", etc. Detection must return None
    # rather than inventing one.
    assert find_contributions(attention.sections) is None


def test_attention_parse_report_clean(attention: ParsedPaper) -> None:
    assert attention.parse_report.warnings == []
    assert attention.parse_report.n_paragraphs == sum(len(s.paragraphs) for s in attention.sections)


# --------------------------------------------------------------------------- 2511.07364


def test_stepwise_header(stepwise: ParsedPaper) -> None:
    assert stepwise.title == (
        "Self-Evaluating LLMs for Multi-Step Tasks: "
        "Stepwise Confidence Estimation for Failure Detection"
    )
    assert "15% relative increase in AUC-ROC" in stepwise.abstract
    assert [a.name for a in stepwise.authors] == ["Vaibhav Mavi", "Shubh Jaroria", "Weiqi Sun"]
    assert all(a.affiliation == "Dyania Health" for a in stepwise.authors)


def test_stepwise_sections(stepwise: ParsedPaper) -> None:
    # 25 sections observed. GROBID merged '2 Related Work' into the intro (it sits after
    # Figure 1's text), merged '4 Data' with '4.1 Agent Inputs' (we split it back), and
    # produced junk headings ('Yes', 'II-III') from the appendix clinical-note figure.
    assert stepwise.parse_report.n_sections == 25
    headings = [(s.number, s.heading, s.level) for s in stepwise.sections]
    assert headings[0] == ("1", "Introduction", 1)
    assert ("4", "Data", 1) in headings
    assert ("4.1", "Agent Inputs", 2) in headings
    # appendix letter numbers recovered from heading text
    assert ("A", "Training details", 1) in headings
    assert ("B.1", "Agent: Training and Inference", 2) in headings
    assert ("C.1.1", "Self-verbalized confidence", 3) in headings


def test_stepwise_references_and_citations(stepwise: ParsedPaper) -> None:
    assert stepwise.parse_report.n_references == 31
    refs = {r.ref_id: r for r in stepwise.references}
    assert refs["b0"].title == "The Internal State of an LLM Knows When It’s Lying"
    assert refs["b0"].doi == "10.18653/v1/2023.findings-emnlp.68"
    assert refs["b0"].url == "https://arxiv.org/abs/2304.13734"
    assert stepwise.parse_report.n_resolved_citations >= 1
    assert stepwise.parse_report.n_unresolved_citations == 0
    cited = {rid for c in stepwise.in_text_citations for rid in c.ref_ids}
    assert cited <= set(refs)


def test_stepwise_tables(stepwise: ParsedPaper) -> None:
    table1 = stepwise.tables[0]
    assert table1.label == "Table 1"
    assert table1.rows is not None
    assert any(row[:3] == ["Regression", "step", "0.907 (+7%)"] for row in table1.rows)


def test_stepwise_contributions_truthfully_absent(stepwise: ParsedPaper) -> None:
    # This workshop paper states its question and findings ("We systematically investigate
    # this question...", "Our experiments reveal...") but has no explicit contributions
    # statement or list, so detection correctly returns None.
    assert find_contributions(stepwise.sections) is None


# --------------------------------------------------------------------------- partial TEI

_TEI_HEAD = '<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader/><text>'


def test_garbage_does_not_crash() -> None:
    paper = parse_tei("not xml at all", "x", "x.pdf")
    assert paper.title == ""
    assert "TEI could not be parsed as XML" in paper.parse_report.warnings


def test_empty_tei_warnings() -> None:
    paper = parse_tei(_TEI_HEAD + "<body/></text></TEI>", "x", "x.pdf")
    warnings = paper.parse_report.warnings
    for expected in ("missing title", "empty abstract", "no sections", "no references"):
        assert expected in warnings


def test_unresolved_and_loose_citations_and_lists() -> None:
    xml = (
        _TEI_HEAD
        + "<body><div><head n='1'>Introduction</head>"
        + "<p>Loose cite <ref type='bibr' target='#b0'>[1]</ref> in a paragraph.</p>"
        + "<p><s>Dangling <ref type='bibr' target='#b9'>[9]</ref>.</s></p>"
        + "<list><item>Item one.</item><item>Item two.</item></list>"
        + "</div><div><p>continued text</p></div></body>"
        + "<back><div type='references'><listBibl>"
        + "<biblStruct xml:id='b0'><monogr><title level='m'>T</title></monogr>"
        + "<note type='raw_reference'>Raw ref.</note></biblStruct>"
        + "</listBibl></div></back></text></TEI>"
    )
    paper = parse_tei(xml, "x", "x.pdf")
    assert len(paper.sections) == 1  # headless div merged into previous section
    assert paper.sections[0].paragraphs == [
        "Loose cite [1] in a paragraph.",
        "Dangling [9].",
        "Item one.",
        "Item two.",
        "continued text",
    ]
    assert [c.ref_ids for c in paper.in_text_citations] == [["b0"], ["b9"]]
    assert paper.parse_report.n_resolved_citations == 1
    assert paper.parse_report.n_unresolved_citations == 1
    assert any("unresolved citation target" in w and "b9" in w for w in paper.parse_report.warnings)


@pytest.mark.parametrize(
    ("heading", "n", "in_back", "expected"),
    [
        ("Introduction", "1", False, [("Introduction", "1")]),
        ("Data 4.1 Agent Inputs", "4", False, [("Data", "4"), ("Agent Inputs", "4.1")]),
        ("B.1 Agent: Training", None, True, [("Agent: Training", "B.1")]),
        ("A Training details", None, True, [("Training details", "A")]),
        ("A Closer Look", None, False, [("A Closer Look", None)]),
        ("Responses:", None, False, [("Responses", None)]),
        ("3.2 Attention", None, False, [("Attention", "3.2")]),
    ],
)
def test_split_heading_number(
    heading: str, n: str | None, in_back: bool, expected: list[tuple[str, str | None]]
) -> None:
    assert split_heading_number(heading, n, in_back) == expected


def test_level_from_number() -> None:
    assert level_from_number(None) == 1
    assert level_from_number("3") == 1
    assert level_from_number("3.2") == 2
    assert level_from_number("C.1.1") == 3


# --------------------------------------------------------------------------- paper ids

SHA = "abcdef0123456789" * 4


def test_paper_id_arxiv() -> None:
    assert paper_id_for(Path("data/papers/arxiv/2410.12345v2.pdf"), SHA) == "2410.12345"
    assert paper_id_for(Path("data/papers/arxiv/1706.03762.pdf"), SHA) == "1706.03762"


def test_paper_id_own() -> None:
    assert paper_id_for(Path("data/papers/own/2410.12345.pdf"), SHA) == "own-2410-12345-abcdef01"
    assert paper_id_for(Path("x/My Great  Paper (v2).pdf"), SHA) == "own-my-great-paper-v2-abcdef01"


def test_slug_truncation() -> None:
    slug = slugify("a" * 39 + "-bbbb")
    assert len(slug) <= 40
    assert not slug.endswith("-")


def test_truncated_tei_does_not_crash() -> None:
    xml = fulltext_tei("1706.03762")
    paper = parse_tei(xml[: len(xml) // 2], "1706.03762", "x.pdf")
    assert paper.title == "Attention Is All You Need"
    assert paper.sections  # the body before the cut survives
    assert "no references" in paper.parse_report.warnings


def test_adjacent_citation_refs_keep_spacing(attention: ParsedPaper) -> None:
    assert "translation [35, 2, 5]." in attention.in_text_citations[0].sentence
