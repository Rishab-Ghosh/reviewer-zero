"""Format checker: headings, declared venue, page counts, anonymization, sections, references.
Offline: synthetic page texts only."""

import pytest

from reviewer.config import Settings
from reviewer.format.checks import (
    Context,
    check_acknowledgments,
    check_author_block,
    check_emails,
    check_page_limits,
    check_pdf_metadata,
    check_sections,
    check_self_citation,
    check_unreferenced,
    check_unresolved,
    check_urls,
    url_identity,
)
from reviewer.format.commands import is_private, report_dir, summary_markdown
from reviewer.format.layout import (
    classify_heading,
    declared_venue,
    find_headings,
    main_text_pages,
    references_span,
)
from reviewer.format.venues import ICML, NEURIPS, VENUES
from reviewer.models import (
    DeclaredVenue,
    FormatCheck,
    FormatSeverity,
    ParsedPaper,
    SubmissionStage,
    Venue,
)

BODY = "\n".join(f"Body line {i} of the paper with ordinary content." for i in range(40))


def _ctx(
    pages: list[str],
    declared: DeclaredVenue | None = None,
    meta: dict | None = None,
    names: list[str] | None = None,
) -> Context:
    headings = find_headings(pages)
    return Context(
        pages=pages,
        metadata=meta or {},
        headings=headings,
        declared=declared or DeclaredVenue(venue=None, stage=SubmissionStage.submission),
        refs_span=references_span(pages, headings),
        author_names=names or [],
    )


NEURIPS_SUB = DeclaredVenue(venue=Venue.neurips, stage=SubmissionStage.submission, year=2026)
NEURIPS_2017 = DeclaredVenue(venue=Venue.neurips, stage=SubmissionStage.camera_ready, year=2017)


# --------------------------------------------------------------------------- layout


@pytest.mark.parametrize(
    ("line", "kind"),
    [
        ("R EFERENCES", "references"),
        ("References", "references"),
        ("6. Acknowledgements", "acknowledgments"),
        ("Acknowledgements We are grateful to Nal", "acknowledgments"),
        ("E THICS S TATEMENT", "ethics"),
        ("R EPRODUCIBILITY S TATEMENT", "reproducibility"),
        ("Appendix", "appendix"),
        ("Appendix A Additional Results", "appendix"),
        ("Appendix B.1.2 contains further details. We leave", None),
        ("In Appendix G we show", None),
        ("Limitations & Future Work. Our results raise", "limitations"),
        ("limitations of prior work are", None),
        ("NeurIPS Paper Checklist", "checklist"),
        ("7 Broader Impacts", "impact"),
        ("LLM Usage Statement", "ai_use"),
        ("References to prior work are given below.", None),
    ],
)
def test_classify_heading(line: str, kind: str | None) -> None:
    assert classify_heading(line) == kind


@pytest.mark.parametrize(
    ("page1", "venue", "stage", "year"),
    [
        (
            "Published as a conference paper at ICLR 2021\nTitle",
            Venue.iclr,
            SubmissionStage.camera_ready,
            2021,
        ),
        (
            "Under review as a conference paper at ICLR 2026",
            Venue.iclr,
            SubmissionStage.submission,
            2026,
        ),
        (
            "31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach",
            Venue.neurips,
            SubmissionStage.camera_ready,
            2017,
        ),
        (
            "Submitted to 40th Conference on Neural Information Processing Systems (NeurIPS 2026).",
            Venue.neurips,
            SubmissionStage.submission,
            2026,
        ),
        (
            "Proceedings of the 41st International Conference on Machine Learning, Vienna",
            Venue.icml,
            SubmissionStage.camera_ready,
            2024,
        ),
        (
            "Workshop on Foo at the 37th Conference on Neural Information Processing Systems "
            "(NeurIPS 2023).",
            None,
            SubmissionStage.preprint,
            None,
        ),
        ("Anonymous Author(s)\nAffiliation", None, SubmissionStage.submission, None),
        ("Preprint. Under review.", None, SubmissionStage.preprint, None),
    ],
)
def test_declared_venue(
    page1: str, venue: Venue | None, stage: SubmissionStage, year: int | None
) -> None:
    d = declared_venue(page1)
    assert (d.venue, d.stage, d.year) == (venue, stage, year)


def test_main_text_pages_per_venue_exclusions() -> None:
    # page 9 ends with an impact statement: excluded for ICML, counted for NeurIPS
    pages = [BODY] * 8 + [
        BODY + "\nImpact Statement\nThis paper presents work.",
        "References\n[1] A.",
    ]
    heads = find_headings(pages)
    icml = main_text_pages(pages, heads, ICML)
    neurips = main_text_pages(pages, heads, NEURIPS)
    assert icml.pages_used == 9 and "Impact Statement" in (icml.ends_at or "")
    assert neurips.pages_used == 9 and "References" in (neurips.ends_at or "")
    assert neurips.main_text_pages == 9.0
    # references at the very top of page 10 -> 9 pages used
    top = main_text_pages(
        [BODY] * 9 + ["References\n[1] A."],
        find_headings([BODY] * 9 + ["References\n[1] A."]),
        NEURIPS,
    )
    assert top.pages_used == 9


def test_page_limit_binding_only_for_declared_current_venue() -> None:
    pages = [BODY] * 11 + ["References\n[1] A."]
    heads = find_headings(pages)
    vps = [main_text_pages(pages, heads, r) for r in VENUES.values()]
    sub = check_page_limits(_ctx(pages, NEURIPS_SUB), vps)
    assert [(f.venue, f.severity) for f in sub] == [(Venue.neurips, FormatSeverity.violation)]
    old = check_page_limits(_ctx(pages, NEURIPS_2017), vps)
    assert [f.severity for f in old] == [FormatSeverity.info]  # 11 > 10 camera, older edition
    undeclared = check_page_limits(
        _ctx(pages, DeclaredVenue(venue=None, stage=SubmissionStage.preprint)), vps
    )
    assert {f.venue for f in undeclared} == set(Venue) and all(
        f.severity == FormatSeverity.info for f in undeclared
    )


# --------------------------------------------------------------------------- anonymization

PAGE1 = (
    "Great Title\nJane Q. Doe\nDepartment of CS, University of Somewhere\njane@somewhere.edu\n"
    "Abstract\nWe do things."
)


def test_author_block_email_and_severity_by_stage() -> None:
    ctx = _ctx([PAGE1, BODY], NEURIPS_SUB, names=["Jane Q. Doe", "Google Brain"])
    [block] = check_author_block(ctx)
    assert block.severity == FormatSeverity.violation and "Jane Q. Doe" in block.evidence[0]
    [email] = check_emails(ctx)
    assert "jane@somewhere.edu" in email.evidence[0]
    cam = _ctx([PAGE1, BODY], NEURIPS_2017, names=["Jane Q. Doe"])
    assert check_author_block(cam)[0].severity == FormatSeverity.info
    anon = _ctx(["Great Title\nAnonymous Author(s)\nAffiliation\nAbstract\nx", BODY], NEURIPS_SUB)
    assert check_author_block(anon) == []


def test_self_citation_patterns() -> None:
    page = BODY + "\nIn our previous work [12] we built X. We previously showed that Y holds."
    found = check_self_citation(_ctx([PAGE1, page], NEURIPS_SUB))
    matched = {f.evidence[1] for f in found}
    assert "matched: our previous work" in matched
    assert "matched: We previously showed" in matched
    assert any("with a citation" in f.detail for f in found)
    assert (
        check_self_citation(_ctx([PAGE1, BODY + "\nIn the previous work of Smith et al. [1]"]))
        == []
    )


@pytest.mark.parametrize(
    ("host", "path", "flag"),
    [
        ("github.com", "/janedoe/repo", True),
        ("github.com", "/anonymous-repo/x", False),
        ("anonymous.4open.science", "/r/abc", False),
        ("janedoe.github.io", "/project", True),
        ("arxiv.org", "/abs/1706.03762", False),
        ("github.com", "", False),
    ],
)
def test_url_identity(host: str, path: str, flag: bool) -> None:
    assert (url_identity(host, path) is not None) == flag


def test_urls_flag_own_links_not_third_party() -> None:
    page = BODY + (
        "\nWe build on the Transformers library (https://github.com/huggingface/transformers)."
        "\n"
        + "filler words " * 30
        + "\nOur code is available at https://github.com/janedoe/coolproj ."
    )
    found = check_urls(_ctx([PAGE1, page], NEURIPS_SUB))
    assert [f.evidence[0] for f in found] == ["p2: https://github.com/janedoe/coolproj"]


def test_acknowledgments_and_metadata() -> None:
    pages = [
        PAGE1,
        BODY + "\nAcknowledgments\nWe thank the NSF for grant 123.",
        "References\n[1] A.",
    ]
    [ack] = check_acknowledgments(_ctx(pages, NEURIPS_SUB))
    assert (
        ack.severity == FormatSeverity.violation
        and ack.source == VENUES[Venue.neurips].acknowledgments_rule.source
    )  # type: ignore[union-attr]
    iclr = DeclaredVenue(venue=Venue.iclr, stage=SubmissionStage.submission, year=2027)
    assert check_acknowledgments(_ctx(pages, iclr))[0].severity == FormatSeverity.info
    [meta] = check_pdf_metadata(_ctx(pages, NEURIPS_SUB, meta={"Author": "Jane Doe"}))
    assert meta.check == FormatCheck.pdf_metadata
    assert check_pdf_metadata(_ctx(pages, NEURIPS_SUB, meta={"Author": "Anonymous"})) == []


# --------------------------------------------------------------------------- sections


def test_required_sections() -> None:
    pages = [PAGE1, BODY, "References\n[1] A."]
    missing = check_sections(_ctx(pages, NEURIPS_SUB))
    req = [f for f in missing if f.check == FormatCheck.required_section]
    assert len(req) == 1 and req[0].severity == FormatSeverity.violation
    with_checklist = [*pages, "NeurIPS Paper Checklist\n1. Claims\nAnswer: [Yes]"]
    assert not [
        f
        for f in check_sections(_ctx(with_checklist, NEURIPS_SUB))
        if f.check == FormatCheck.required_section
    ]
    old = check_sections(_ctx(pages, NEURIPS_2017))
    assert all(f.severity == FormatSeverity.info for f in old)


# --------------------------------------------------------------------------- references


def test_unresolved_markers_outside_reference_list() -> None:
    pages = [PAGE1, BODY + "\nas shown in Section ?? and [?].", "References\n[1] What?? A title."]
    [f] = check_unresolved(_ctx(pages))
    assert "2 unresolved" in f.detail


def test_unreferenced_figures_and_tables() -> None:
    page = (
        "Figure 1: Overview of the method.\n"
        "Table 1: Results.\nTable 2. Hyperparameters.\nTable 4.2 summarizes results on MNIST.\n"
        "As shown in fig. 1 and Tables 1-2, it works.\n"
        "Figure 3: Samples.\nFigure 11, 13, and 3 show samples.\n"
        "Figure 2: Never mentioned anywhere.\n"
    )
    found = check_unreferenced(_ctx([PAGE1, page]))
    assert [f.detail for f in found] == ["Figure 2 is never referenced in the text"]


# --------------------------------------------------------------------------- routing / summary


def test_private_routing(tmp_path) -> None:
    s = Settings(_env_file=None, data_dir=tmp_path / "data", reports_dir=tmp_path / "reports")
    own = ParsedPaper(paper_id="own-x-deadbeef", source_path="data/papers/own/x.pdf", title="T")
    pub = ParsedPaper(
        paper_id="1706.03762", source_path="data/papers/arxiv/1706.03762.pdf", title="T"
    )
    openreview = ParsedPaper(
        paper_id="or-AbC123", source_path="data/openreview/pdf/AbC123.pdf", title="T"
    )
    assert is_private(own) and not is_private(pub)
    assert is_private(openreview)
    assert report_dir(s, True) == tmp_path / "reports" / "private" / "format"
    assert report_dir(s, False) == tmp_path / "reports" / "format"


def test_summary_renders_counts_and_sample() -> None:
    from datetime import datetime

    from reviewer.models import FormatReport

    pages = [PAGE1, BODY + "\nFigure 9: Lonely figure.", "References\n[1] A."]
    ctx = _ctx(pages, NEURIPS_SUB, names=["Jane Q. Doe"])
    heads = ctx.headings
    vps = [main_text_pages(pages, heads, r) for r in VENUES.values()]
    from reviewer.format.checks import run_all

    rep = FormatReport(
        paper_id="p1",
        title="T",
        private=False,
        generated=datetime(2026, 9, 25),
        n_pages=3,
        declared=NEURIPS_SUB,
        pages=vps,
        findings=run_all(ctx, vps),
    )
    text = summary_markdown([rep], seed=7, n_sample=10)
    assert "| p1 | NeurIPS submission |" in text and "random findings" in text


def test_toc_and_wrapped_appendix_mentions_are_not_headings() -> None:
    toc = "Contents\n" + "\n".join(f"{i} Section {i} . . . . . . . . {i + 2}" for i in range(1, 8))
    toc += "\nAppendices . . . . . . . . 20"
    wrapped = BODY + "\nFurther details are given in\nAppendix A.\nMore body text follows."
    pages = [PAGE1, toc, wrapped, BODY, "References\n[1] A."]
    kinds = [(h.kind, h.page) for h in find_headings(pages)]
    assert kinds == [("references", 5)]


def test_urls_built_on_third_party_code_are_not_own() -> None:
    page = BODY + (
        "\nOur implementation uses Apex's FMHA code (https://github.com/NVIDIA/apex) as a start."
        "\n"
        + "filler words " * 30
        + "\nOur implementation is based on the open-sourced code in https://github.com/abc/def."
    )
    assert check_urls(_ctx([PAGE1, page], NEURIPS_SUB)) == []
