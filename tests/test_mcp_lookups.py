import json

import anyio
import httpx
import pytest
from mcp.client import Client
from pybtex.database import parse_string

from reviewer.mcp.config import McpConfig
from reviewer.mcp.index import HostedIndex
from reviewer.mcp.lookups import run_citations, run_paper
from reviewer.mcp.prior_work import QueryError
from reviewer.mcp.server import build_server

LORA = {
    "key": "2106.09685",
    "kind": "arxiv",
    "title": "LoRA: Low-Rank Adaptation of Large Language Models",
    "authors": ["Edward J. Hu", "Yelong Shen"],
    "year": 2021,
    "venue": "ICLR",
    "doi": None,
    "arxiv_id": "2106.09685",
    "abstract": "We propose LoRA.",
}
CITED = [
    {
        "key": f"17{i:02d}.00001",
        "kind": "arxiv",
        "title": f"Cited work number {i} on transformers",
        "year": 2017,
        "relation": "cites",
    }
    for i in range(5)
]


def handler(req: httpx.Request) -> httpx.Response:
    path = req.url.path
    if path == "/paper/2106.09685":
        return httpx.Response(200, json=LORA)
    if path.startswith("/paper/") or path.startswith("/citations/9"):
        return httpx.Response(404, json={"error": "not_found", "detail": "Paper not found."})
    if path == "/citations/2106.09685":
        assert req.url.params["direction"] == "cites"
        return httpx.Response(200, json=CITED[: int(req.url.params["limit"])])
    if path == "/papers":
        keys = json.loads(req.content)["keys"]
        rows = [
            {
                "key": k,
                "title": f"Detail {k}",
                "year": 2017,
                "authors": ["A Vaswani"],
                "arxiv_id": k,
            }
            for k in keys
            if k != "1704.00001"
        ]
        return httpx.Response(200, json={"papers": rows, "missing": ["1704.00001"]})
    return httpx.Response(404, json={"error": "not_found", "detail": "Not Found"})


@pytest.fixture
def index(tmp_path) -> HostedIndex:
    return HostedIndex(
        McpConfig(index_url="https://index.test", index_key="k", cache=tmp_path),
        transport=httpx.MockTransport(handler),
        sleep=lambda s: None,
    )


def test_paper_returns_metadata_link_and_bibtex(index):
    p = run_paper(index, " 2106.09685 ")
    assert p.title.startswith("LoRA") and p.authors == ["Edward J. Hu", "Yelong Shen"]
    assert p.link == "https://arxiv.org/abs/2106.09685"
    assert p.bibtex.startswith("@inproceedings{hu2021lora,")
    assert len(parse_string(p.bibtex, "bibtex").entries) == 1


def test_unknown_or_malformed_key_is_a_clear_error(index):
    with pytest.raises(QueryError, match=r"No paper with key '9999\.99999'"):
        run_paper(index, "9999.99999")
    with pytest.raises(QueryError, match="one paper key"):
        run_paper(index, "two keys")
    with pytest.raises(QueryError, match="No paper with key"):
        run_citations(index, "9999.99999")


def test_citations_with_metadata_truncation_and_fallback_details(index):
    out = run_citations(index, "2106.09685", "cites", limit=3)
    assert out.truncated and [p.key for p in out.papers] == [
        "1700.00001",
        "1701.00001",
        "1702.00001",
    ]
    assert all(p.authors == ["A Vaswani"] for p in out.papers)
    keys = [next(iter(parse_string(p.bibtex, "bibtex").entries)) for p in out.papers]
    assert len(set(keys)) == 3  # collision suffixes keep keys unique
    full = run_citations(index, "2106.09685", "cites", limit=50)
    assert not full.truncated and len(full.papers) == 5
    missing = next(p for p in full.papers if p.key == "1704.00001")
    assert missing.title == "Cited work number 4 on transformers" and missing.authors is None
    with pytest.raises(QueryError, match="limit must be between 1 and 200"):
        run_citations(index, "2106.09685", limit=500)


def test_lookups_through_an_mcp_client(index):
    async def go():
        async with Client(build_server(index_factory=lambda: index)) as c:
            names = {t.name for t in (await c.list_tools()).tools}
            ok = await c.call_tool("paper", {"key": "2106.09685"})
            cit = await c.call_tool("citations", {"key": "2106.09685", "limit": 2})
            bad = await c.call_tool("paper", {"key": "9999.99999"})
            wrong_dir = await c.call_tool("citations", {"key": "2106.09685", "direction": "both"})
            return names, ok, cit, bad, wrong_dir

    names, ok, cit, bad, wrong_dir = anyio.run(go)
    assert {"find_prior_work", "paper", "citations"} <= names
    assert not ok.is_error and ok.structured_content["key"] == "2106.09685"
    assert len(cit.structured_content["papers"]) == 2
    assert bad.is_error and "No paper with key" in bad.content[0].text
    assert wrong_dir.is_error
