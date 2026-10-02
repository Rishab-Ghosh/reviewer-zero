import shutil
from pathlib import Path

import anyio
import pytest
from mcp.client import Client

import reviewer.mcp.format_tool as ft
from reviewer.format.checks import Context, run_all
from reviewer.format.commands import check_pdf
from reviewer.format.layout import find_headings, main_text_pages, references_span
from reviewer.format.venues import VENUES
from reviewer.mcp.config import McpConfig
from reviewer.mcp.format_tool import run_check_format
from reviewer.mcp.local_parse import grobid_paper, is_local_url
from reviewer.mcp.prior_work import QueryError
from reviewer.mcp.server import build_server
from reviewer.models import (
    Author,
    DeclaredVenue,
    FormatCheck,
    FormatSeverity,
    ParsedPaper,
    SubmissionStage,
)

DEMO = Path(__file__).resolve().parents[1] / "docs" / "demo" / "spindle.pdf"
OTHER_PAGES = [
    "A Method For Things\nJane Roe\nDepartment of Computer Science, Example University\n"
    "Abstract\nWe study things. Code: https://github.com/janeroe/things\n"
    "1 Introduction\nIn our previous work [3] we showed this.\n",
    "2 Method\nText.\nAcknowledgments\nWe thank the Example Foundation for funding.\n"
    "References\n[1] A. Author. A paper. 2020.\n",
]


def _other_ctx(pages: list[str]) -> Context:
    headings = find_headings(pages)
    return Context(
        pages=pages,
        metadata={"Author": "Jane Roe"},
        headings=headings,
        declared=DeclaredVenue(venue=None, stage=SubmissionStage.submission),
        refs_span=references_span(pages, headings),
        author_names=["Jane Roe"],
        rules_mode="none",
    )


def test_d18_other_runs_anonymization_without_venue_rules_or_sources():
    ctx = _other_ctx(OTHER_PAGES)
    pages = [main_text_pages(ctx.pages, ctx.headings, r) for r in VENUES.values()]
    findings = run_all(ctx, pages)
    checks = {f.check for f in findings}
    author = [f for f in findings if f.check == FormatCheck.author_block]
    assert author and author[0].severity == FormatSeverity.violation
    assert FormatCheck.pdf_metadata in checks
    assert not checks & {
        FormatCheck.page_limit,
        FormatCheck.required_section,
        FormatCheck.recommended_section,
    }
    acks = [f for f in findings if f.check == FormatCheck.acknowledgments]
    assert acks and acks[0].severity == FormatSeverity.info
    assert all(f.source is None for f in findings)


def test_declared_mode_still_cites_sources():
    ctx = _other_ctx(OTHER_PAGES)
    ctx = Context(**{**ctx.__dict__, "rules_mode": "declared"})
    pages = [main_text_pages(ctx.pages, ctx.headings, r) for r in VENUES.values()]
    assert any(f.source for f in run_all(ctx, pages))


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="poppler not installed")
def test_pypdf_fallback_matches_pdftotext_on_the_demo_pdf():
    def findings(engine: str) -> list[tuple]:
        r, doc = check_pdf(
            DEMO, paper_id="d", title="", private=False, author_names=[], engine=engine
        )
        assert doc.engine == engine
        return sorted((f.check.value, f.severity.value, f.page) for f in r.findings)

    assert findings("pdftotext") == findings("pypdf")


def _grobid(monkeypatch, paper: ParsedPaper | None) -> None:
    monkeypatch.setattr(ft, "grobid_paper", lambda pdf, config: paper)


def test_tool_with_grobid_names_the_authors(monkeypatch, tmp_path):
    paper = ParsedPaper(
        paper_id="p",
        source_path=str(DEMO),
        title="SPINDLE",
        authors=[Author(name="Nemo Quillfeather"), Author(name="Ada Placeholder")],
    )
    _grobid(monkeypatch, paper)
    r = run_check_format(str(DEMO), "iclr2027", "submission", McpConfig(cache=tmp_path))
    assert r.author_names_from == "grobid" and r.rules == "ICLR 2027"
    block = next(f for f in r.findings if f.check == FormatCheck.author_block)
    assert block.severity == FormatSeverity.violation
    assert "Nemo Quillfeather" in " ".join(block.evidence)
    assert r.page_limit is not None and r.main_text_pages is not None
    assert not any("does not match" in n for n in r.notes)


def test_tool_without_grobid_says_so_and_other_venue_has_no_rules(monkeypatch, tmp_path):
    _grobid(monkeypatch, None)
    r = run_check_format(str(DEMO), "other", "submission", McpConfig(cache=tmp_path))
    assert r.author_names_from == "none"
    assert any("GROBID is not running" in n for n in r.notes)
    assert r.main_text_pages is None and r.page_limit is None
    assert all(f.source is None for f in r.findings)
    assert any(f.check == FormatCheck.author_block for f in r.findings)  # affiliation line
    assert any("does not match" in n for n in r.notes)  # the PDF says ICLR 2027


def test_tool_warns_on_venue_mismatch(monkeypatch, tmp_path):
    _grobid(monkeypatch, None)
    r = run_check_format(str(DEMO), "neurips2026", "submission", McpConfig(cache=tmp_path))
    assert r.detected_header.text and "ICLR" in r.detected_header.text
    assert any("does not match the venue you chose (neurips2026)" in n for n in r.notes)


def test_bad_inputs(tmp_path):
    with pytest.raises(QueryError, match="No file"):
        run_check_format(str(tmp_path / "nope.pdf"), "iclr2027", "submission")
    txt = tmp_path / "x.pdf"
    txt.write_text("not a pdf")
    with pytest.raises(QueryError, match="is not a PDF"):
        run_check_format(str(txt), "iclr2027", "submission")


def test_grobid_only_at_loopback(monkeypatch, tmp_path):
    assert is_local_url("http://localhost:8070") and is_local_url("http://127.0.0.1:8070")
    assert not is_local_url("https://grobid.example.com")
    monkeypatch.setenv("GROBID_URL", "https://grobid.example.com")
    assert grobid_paper(DEMO, McpConfig(cache=tmp_path)) is None  # never sent there


def test_check_format_through_an_mcp_client(monkeypatch, tmp_path):
    _grobid(monkeypatch, None)
    monkeypatch.setenv("REVIEWER_ZERO_CACHE", str(tmp_path))

    async def go():
        async with Client(build_server()) as c:
            ok = await c.call_tool(
                "check_format",
                {"pdf_path": str(DEMO), "venue": "iclr2027", "stage": "submission"},
            )
            bad = await c.call_tool(
                "check_format",
                {
                    "pdf_path": str(tmp_path / "missing.pdf"),
                    "venue": "iclr2027",
                    "stage": "submission",
                },
            )
            return ok, bad

    ok, bad = anyio.run(go)
    assert not ok.is_error and ok.structured_content["rules"] == "ICLR 2027"
    assert bad.is_error and "No file" in bad.content[0].text
