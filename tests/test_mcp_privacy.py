"""MCP_PLAN §5: every tool contacts only the allowed hosts, and no request carries the paper's
body text. Every httpx request is recorded (and answered locally); arXiv's curl is recorded too."""

import re
from contextlib import suppress
from pathlib import Path
from urllib.parse import unquote_plus, urlsplit

import httpx
import pytest

import reviewer.mcp.format_tool as ft
from reviewer.citations.arxiv_meta import ArxivMetaClient
from reviewer.format.pdf import load_pdf
from reviewer.mcp.citations_tool import run_check_citations
from reviewer.mcp.config import McpConfig
from reviewer.mcp.format_tool import run_check_format
from reviewer.mcp.index import HostedIndex
from reviewer.mcp.lookups import run_citations, run_paper
from reviewer.mcp.prior_work import QueryError, run_find_prior_work
from reviewer.mcp.quotes_tool import run_verify_quotes
from reviewer.mcp.review_tool import run_review_paper
from reviewer.parse.grobid_client import GrobidClient

DEMO = Path(__file__).resolve().parents[1] / "docs" / "demo" / "spindle.pdf"
INDEX = "https://index.test"
ALLOWED = {"index.test", "api.openalex.org", "api.crossref.org", "export.arxiv.org"}
QUERIES = [
    f"a frozen pretrained model is adapted by learning a small low rank update number {i}"
    for i in range(6)
]
HIT = {
    "key": "2106.09685",
    "kind": "arxiv",
    "title": "LoRA: Low-Rank Adaptation of Large Language Models",
    "abstract": "We propose low-rank adaptation.",
    "published_date": "2021-06-17",
    "year": 2021,
    "similarity": 0.8,
}


class Recorder:
    def __init__(self) -> None:
        self.requests: list[tuple[str, str]] = []  # (decoded url, decoded body)

    def answer(self, req: httpx.Request) -> httpx.Response:
        self.requests.append(
            (unquote_plus(str(req.url)), unquote_plus(req.content.decode("utf-8", "replace")))
        )
        path = req.url.path
        if req.url.host != "index.test":
            return httpx.Response(404, json={}, request=req)
        if path == "/health":
            body = {"index_version": "t"}
        elif path == "/search":
            body = [HIT]
        elif path == "/expand":
            body = {"candidates": [], "total": 0}
        elif path == "/papers":
            body = {"papers": [], "missing": []}
        else:
            return httpx.Response(
                404, json={"error": "not_found", "detail": "Paper not found."}, request=req
            )
        return httpx.Response(200, json=body, request=req)


@pytest.fixture
def recorder(monkeypatch, tmp_path):
    rec = Recorder()
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda self, req: rec.answer(req))

    def arxiv(self, url):  # replaces the curl call and its retries
        rec.requests.append((unquote_plus(url), ""))

    monkeypatch.setattr(ArxivMetaClient, "_fetch", arxiv)
    monkeypatch.setattr(GrobidClient, "is_alive", lambda self: False)
    monkeypatch.setattr(ft, "grobid_paper", lambda pdf, config: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("GROBID_URL", "http://localhost:8070")
    return rec, McpConfig(index_url=INDEX, index_key="k", cache=tmp_path, index_retries=0)


def _body_sentences() -> list[str]:
    """Sentences of the demo paper's body (before the reference list), lowercased."""
    text = "\n".join(load_pdf(DEMO).pages)
    body = re.split(r"\n\s*References\s*\n", text)[0]
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(body.split()))
    return [s.lower()[:60] for s in sentences if len(s) >= 60]


def test_every_tool_contacts_only_allowed_hosts_and_never_sends_body_text(recorder):
    rec, config = recorder
    index = HostedIndex(config)
    run_check_format(str(DEMO), "iclr2027", "submission", config)
    run_verify_quotes(str(DEMO), ["the paper says something that is not there"], config)
    run_find_prior_work(index, QUERIES, None, 40)
    for lookup in (run_paper, lambda i, k: run_citations(i, k, "cites", 10)):
        with suppress(QueryError):  # the recorder knows no papers
            lookup(index, "2106.09685")
    run_check_citations(str(DEMO), config)
    with pytest.raises(QueryError, match="docker run"):  # no local GROBID: refuses before any call
        run_review_paper(str(DEMO), "iclr2027", confirm=True, config=config)

    hosts = {urlsplit(url).hostname for url, _ in rec.requests}
    assert hosts <= ALLOWED, hosts - ALLOWED
    assert "index.test" in hosts and "api.openalex.org" in hosts  # the tools really ran
    sentences = _body_sentences()
    assert len(sentences) > 20
    sent = "\n".join(f"{url}\n{body}".lower() for url, body in rec.requests)
    leaked = [s for s in sentences if s in sent]
    assert not leaked, leaked[:3]


def test_index_key_goes_only_to_the_index(recorder, monkeypatch):
    rec, config = recorder
    seen: list[tuple[str, str | None]] = []

    def answer(self, req):
        seen.append((req.url.host, req.headers.get("x-api-key")))
        return rec.answer(req)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", answer)
    run_check_citations(str(DEMO), config)
    assert {h for h, k in seen if k} == {"index.test"}
