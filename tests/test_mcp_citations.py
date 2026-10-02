from datetime import UTC, datetime
from pathlib import Path

import anyio
import httpx
import pytest
from mcp.client import Client

import reviewer.citations.commands as cc
import reviewer.mcp.citations_tool as ct
from reviewer.citations import raw as rawlib
from reviewer.citations.fallback_refs import (
    fallback_references,
    reference_lines,
    split_references,
    until_list_ends,
)
from reviewer.config import Settings
from reviewer.format.pdf import load_pdf
from reviewer.mcp.config import McpConfig
from reviewer.mcp.index import HostedIndex, IndexAuthError
from reviewer.mcp.server import build_server
from reviewer.models import (
    CitationReport,
    InTextCitation,
    ParsedPaper,
    Reference,
    ReferenceCheck,
)

DEMO = Path(__file__).resolve().parents[1] / "docs" / "demo" / "spindle.pdf"


def test_numbered_bracket_list():
    lines = [
        "[1] A. Author. First paper title here. In Venue, 2020.",
        "continued line.",
        "[2] B. Writer. Second title of a paper. arXiv, 2021.",
        "[3] C. Person. Third title words here. J. Things, 2019.",
    ]
    out = split_references(lines)
    assert len(out) == 3 and out[0].endswith("continued line.")


def test_numbered_mode_needs_the_list_to_start_at_entry_one():
    lines = [
        "Ashish Vaswani, Noam Shazeer. Attention is all you need. In NeurIPS, 2017.",
        "1. Step one of the appendix algorithm.",
        "2. Step two.",
        "3. Step three.",
    ]
    assert split_references(lines)[0].startswith("Ashish Vaswani")


def test_author_year_starts_hyphens_and_ligatures():
    lines = [
        "Neil Houlsby, Andrei Giurgiu, and Sylvain Gelly. Parameter-eﬀicient trans-",
        "fer learning for NLP. In International Conference on Machine",
        "Learning, 2019.",
        "Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts.",
        "Elad Ben Zaken, Shauli Ravfogel, and Yoav Goldberg. BitFit: Simple parameter-",
        "efficient fine-tuning. In Proceedings of ACL, 2022.",
        "Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In",
        "International Conference on Learning Representations, 2019.",
    ]
    out = split_references(lines)
    assert [e.split(",")[0].split(" and ")[0] for e in out] == [
        "Neil Houlsby",
        "Xiang Lisa Li",
        "Elad Ben Zaken",
        "Ilya Loshchilov",
    ]
    assert "Parameter-efficient transfer learning" in out[0]  # ligature + hyphen repaired
    assert rawlib.raw_title(out[0]) == "Parameter-efficient transfer learning for NLP"


def test_list_ends_at_a_run_of_non_references():
    refs = ["A. Author, B. Other. A title of a paper. In Venue, 2020."] * 2
    junk = ["Receiver head ablation: we ablated the top heads.", "Algorithm 1 step.", "Table 4."]
    assert len(until_list_ends(refs + junk + refs)) == 2


def test_running_headers_and_page_numbers_are_dropped():
    pages = [
        "Running Header 2026\nTitle\n1 Introduction\nText.\n1",
        "Running Header 2026\nMore text.\n2",
        "Running Header 2026\nReferences\nA. Author, B. Other. A title of a paper. In V, 2020.\n3",
        "Running Header 2026\nC. Third, D. Fourth. Another paper title. In W, 2021.\n4",
    ]
    lines = reference_lines(pages)
    assert lines and not any(ln.startswith("Running Header") or ln.isdigit() for ln in lines)


def test_demo_pdf_gives_all_nine_references():
    refs = fallback_references(load_pdf(DEMO).pages)
    assert len(refs) == 9 and refs[0].raw_string.startswith("Ashish Vaswani")


def test_support_false_makes_no_llm_call(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("support judge called")

    monkeypatch.setattr(cc, "judge_sentence", boom)
    s = Settings(_env_file=None, anthropic_api_key=None, index_url="http://127.0.0.1:9")
    paper = ParsedPaper(
        paper_id="p",
        source_path="p.pdf",
        title="T",
        in_text_citations=[InTextCitation(section_heading="1", sentence="X [1].", ref_ids=["b0"])],
    )
    report = cc.Checker(s).check(paper, support=False)
    assert report.support == [] and report.token_usage.cost_usd == 0
    with pytest.raises(AssertionError, match="support judge called"):
        cc.Checker(s).check(paper, support=True)


def _fake_report(paper: ParsedPaper) -> CitationReport:
    checks = [ReferenceCheck(ref_id=r.ref_id, raw_string=r.raw_string) for r in paper.references]
    return CitationReport(
        paper_id=paper.paper_id,
        title="",
        private=True,
        generated=datetime.now(UTC),
        index_available=True,
        model="m",
        prompt_version="v",
        references=checks,
    )


@pytest.fixture
def no_network(monkeypatch):
    calls = {}

    def fake_check(self, paper, support=True):
        calls["support"] = support
        calls["n_refs"] = len(paper.references)
        return _fake_report(paper)

    monkeypatch.setattr(ct.Checker, "check", fake_check)
    monkeypatch.setattr(HostedIndex, "paper", lambda self, key: None)
    return calls


def test_tool_falls_back_without_grobid(monkeypatch, tmp_path, no_network):
    monkeypatch.setattr(ct, "grobid_paper", lambda pdf, config: None)
    cfg = McpConfig(index_url="https://index.test", index_key="k", cache=tmp_path)
    r = ct.run_check_citations(str(DEMO), cfg)
    assert r.parser == "fallback" and r.text_engine in ("pdftotext", "pypdf")
    assert r.n_references == 9 and no_network == {"support": False, "n_refs": 9}
    assert any("GROBID is not running" in n for n in r.notes)


def test_tool_uses_grobid_references_when_available(monkeypatch, tmp_path, no_network):
    paper = ParsedPaper(
        paper_id="p",
        source_path=str(DEMO),
        title="T",
        references=[
            Reference(ref_id="b0", raw_string="A. Author. A title of a paper. In V, 2020.")
        ],
    )
    monkeypatch.setattr(ct, "grobid_paper", lambda pdf, config: paper)
    cfg = McpConfig(index_url="https://index.test", index_key="k", cache=tmp_path)
    r = ct.run_check_citations(str(DEMO), cfg)
    assert r.parser == "grobid" and r.text_engine is None and r.n_references == 1


def test_bad_key_reaches_the_model_as_a_clear_error(monkeypatch, tmp_path):
    def refuse(self, key):
        raise IndexAuthError("The Reviewer Zero index refused the API key (/paper).")

    monkeypatch.setattr(HostedIndex, "paper", refuse)
    monkeypatch.setenv("REVIEWER_ZERO_INDEX_URL", "https://index.test")
    monkeypatch.setenv("REVIEWER_ZERO_CACHE", str(tmp_path))

    async def go():
        async with Client(build_server()) as c:
            return await c.call_tool("check_citations", {"pdf_path": str(DEMO)})

    res = anyio.run(go)
    assert res.is_error and "refused the API key" in res.content[0].text


def test_index_key_header_reaches_the_citation_sources(tmp_path):
    from reviewer.citations.sources import IndexClient

    seen = []
    s = Settings(
        _env_file=None, index_url="https://index.test", index_api_key="k123", cache_root=tmp_path
    )
    ic = IndexClient(s)
    ic._client("ns")._http._transport = httpx.MockTransport(
        lambda req: seen.append(req.headers.get("X-API-Key")) or httpx.Response(200, json=[])
    )
    ic._available = True
    ic.search("a title", 3)
    assert seen == ["k123"]


def test_an_overlong_entry_ends_the_list():
    ref = "A. Author, B. Other. A title of a paper. In Venue, 2020."
    glued = ref + " Appendix text continues here. " * 80
    assert until_list_ends([ref, glued, ref]) == [ref]
