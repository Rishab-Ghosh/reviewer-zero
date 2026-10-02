from pathlib import Path

import anyio
import pytest
from mcp.client import Client

import reviewer.mcp.quotes_tool as qt
from reviewer.format.pdf import load_pdf
from reviewer.mcp.config import McpConfig
from reviewer.mcp.prior_work import QueryError
from reviewer.mcp.quotes_tool import run_verify_quotes
from reviewer.mcp.server import build_server

DEMO = Path(__file__).resolve().parents[1] / "docs" / "demo" / "spindle.pdf"
REAL = (
    "Fine-tuning every weight of a large pretrained language model is expensive to train "
    "and to store."
)


@pytest.fixture(autouse=True)
def no_grobid(monkeypatch):
    monkeypatch.setattr(qt, "grobid_paper", lambda pdf, config: None)


def _check(tmp_path, quotes):
    return run_verify_quotes(str(DEMO), quotes, McpConfig(cache=tmp_path))


def test_verbatim_quote_is_found_with_its_page(tmp_path):
    r = _check(tmp_path, [REAL, REAL.replace("expensive", "cheap"), "a large pretrained model"])
    assert [(c.status, c.pages) for c in r.quotes] == [
        ("verified", [1]),
        ("not_found", []),
        ("too_short", []),
    ]
    assert r.n_verified == 1


def test_ellipsis_fragments_must_appear_in_order(tmp_path):
    ok = "Fine-tuning every weight of a large ... expensive to train and to store."
    reversed_ = "expensive to train and to store ... Fine-tuning every weight of a large"
    r = _check(tmp_path, [ok, reversed_])
    assert [c.status for c in r.quotes] == ["verified", "not_found"]


def test_quote_across_a_page_break(tmp_path):
    pages = load_pdf(DEMO).pages
    tail = " ".join(pages[0].split()[-4:])
    head = " ".join(pages[1].split()[:4])
    r = _check(tmp_path, [f"{tail} {head}"])
    assert r.quotes[0].status == "verified" and r.quotes[0].pages == [1, 2]


def test_input_limits(tmp_path):
    with pytest.raises(QueryError, match="1 to 50"):
        _check(tmp_path, [])
    with pytest.raises(QueryError, match="at most 1000"):
        _check(tmp_path, ["word " * 300])


def test_verify_quotes_through_an_mcp_client(tmp_path, monkeypatch):
    monkeypatch.setenv("REVIEWER_ZERO_CACHE", str(tmp_path))

    async def go():
        async with Client(build_server()) as c:
            return await c.call_tool("verify_quotes", {"pdf_path": str(DEMO), "quotes": [REAL]})

    res = anyio.run(go)
    assert not res.is_error and res.structured_content["n_verified"] == 1
