import json
from datetime import date

import anyio
import httpx
import pytest
from mcp.client import Client
from pybtex.database import parse_string
from pybtex.plugin import find_plugin

from reviewer.config import Settings
from reviewer.mcp.bibtex import bibtex, cite_key, unique_keys
from reviewer.mcp.config import McpConfig
from reviewer.mcp.index import HostedIndex
from reviewer.mcp.prior_work import QueryError, run_find_prior_work, validate
from reviewer.mcp.server import build_server
from reviewer.models import IndexPaperDetail, SearchQuery
from reviewer.novelty.firststage import (
    LogisticModel,
    feature_matrix,
    gather_candidates,
    select_pool,
)
from reviewer.novelty.retrieve import Target

QUERIES = [
    f"we freeze the pretrained weights and learn a low rank update component number {i}"
    for i in range(1, 8)
]
WORDS = ["adaptation", "projection", "rank", "frozen", "update", "matrix", "scaling"]


def _paper(j: int, sim: float) -> dict:
    return {
        "key": f"2001.{j:05d}",
        "kind": "arxiv",
        "title": f"A study of {WORDS[j % 7]} methods number {j} for transformers",
        "abstract": f"Abstract {j}. " + "x" * 2000,
        "published_date": "2020-01-15",
        "year": 2020,
        "similarity": sim,
    }


class FakeIndexServer:
    def __init__(self) -> None:
        self.expand_bodies: list[dict] = []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if path == "/health":
            return httpx.Response(200, json={"index_version": "fixture-v1"})
        if path == "/search":
            i = QUERIES.index(req.url.params["q"])
            assert req.url.params["before"] == "2025-06-01"
            hits = [_paper(i * 4 + r, 0.9 - 0.01 * r) for r in range(int(req.url.params["k"]))]
            return httpx.Response(200, json=hits)
        if path == "/expand":
            body = json.loads(req.content)
            self.expand_bodies.append(body)
            batch = len(self.expand_bodies)
            cands = [
                {
                    **_paper(900 + r, None),
                    "seed_count": 3 + r,
                    "max_cosine": 0.7 + 0.01 * r,
                    "seeds": [{"seed": body["seeds"][0], "relation": "cites"}],
                }
                for r in range(5)
            ]
            # the same expansion-only key in both batches: batch 2 has the higher cosine
            cands.append(
                {
                    **_paper(999, None),
                    "seed_count": batch,
                    "max_cosine": 0.5 + batch / 10,
                    "seeds": [{"seed": body["seeds"][batch], "relation": "cited_by"}],
                }
            )
            return httpx.Response(200, json={"candidates": cands, "total": len(cands)})
        if path == "/papers":
            keys = json.loads(req.content)["keys"]
            papers = [
                {
                    "key": k,
                    "title": f"Detail {k}",
                    "year": 2020,
                    "venue": "NeurIPS",
                    "authors": ["Ada Lovelace", "Alan Turing"],
                    "arxiv_id": k,
                    "doi": None,
                }
                for k in keys
                if k != "2001.00999"
            ]
            missing = [k for k in keys if k == "2001.00999"]
            return httpx.Response(200, json={"papers": papers, "missing": missing})
        return httpx.Response(404, json={"error": "not_found", "detail": "Not Found"})


@pytest.fixture
def server() -> FakeIndexServer:
    return FakeIndexServer()


@pytest.fixture
def index(tmp_path, server) -> HostedIndex:
    return HostedIndex(
        McpConfig(index_url="https://index.test", index_key="k", cache=tmp_path),
        transport=httpx.MockTransport(server),
        sleep=lambda s: None,
    )


BEFORE = date(2025, 6, 1)


def test_order_and_scores_equal_the_pipelines_first_stage(index):
    out = run_find_prior_work(index, QUERIES, BEFORE, k=40)
    sq = [
        SearchQuery(id=f"q{i}", origin="claim", kind="component", text=q)
        for i, q in enumerate(QUERIES, 1)
    ]
    from reviewer.mcp.prior_work import PARAMS

    cands, _ = gather_candidates(sq, index, Target(cutoff=BEFORE), PARAMS)
    model = LogisticModel.load(Settings(_env_file=None).first_stage_model_file)
    scores = model(feature_matrix(cands))
    expected = select_pool(cands, scores, 40)
    assert [c.key for c in out.candidates] == [cands[i].candidate.paper.key for i in expected]
    assert [c.score for c in out.candidates] == [round(float(scores[i]), 4) for i in expected]
    assert len(out.candidates) == 40 and out.n_considered == len(cands) > 40
    assert [c.rank for c in out.candidates] == list(range(1, 41))
    assert out.index_version == "fixture-v1"


def test_expand_is_batched_by_five_and_merged(index, server):
    out = run_find_prior_work(index, QUERIES, BEFORE, k=40)
    assert [len(b["queries"]) for b in server.expand_bodies] == [5, 2]
    assert all(b["max_candidates"] <= 500 for b in server.expand_bodies)
    server.expand_bodies.clear()
    merged = index.expand(["2001.00000", "2001.00001", "2001.00002"], QUERIES, BEFORE, 2000)
    dup = next(c for c in merged.candidates if c.key == "2001.00999")
    assert dup.max_cosine == pytest.approx(0.7) and dup.seed_count == 2 and len(dup.seeds) == 2
    assert out.queries == QUERIES


def test_candidate_fields(index):
    out = run_find_prior_work(index, QUERIES, BEFORE, k=40)
    by_key = {c.key: c for c in out.candidates}
    first = by_key["2001.00000"]  # only query 1 returns it
    assert first.matched_queries == [1]
    assert first.authors == ["Ada Lovelace", "Alan Turing"] and first.venue == "NeurIPS"
    assert first.link == "https://arxiv.org/abs/2001.00000"
    assert len(first.abstract) == 1200
    expansion_only = [c for c in out.candidates if c.key.startswith("2001.009")]
    assert expansion_only and all(c.matched_queries == [] for c in expansion_only)
    assert all(c.seed_count >= 1 for c in expansion_only)


def test_every_bibtex_entry_parses_and_formats(index):
    out = run_find_prior_work(index, QUERIES, BEFORE, k=40)
    db = parse_string("\n\n".join(c.bibtex for c in out.candidates), "bibtex")
    assert len(db.entries) == 40
    style = find_plugin("pybtex.style.formatting", "plain")()
    rendered = list(style.format_entries(db.entries.values()))
    assert len(rendered) == 40 and all(e.text.render_as("text") for e in rendered)


def test_bibtex_builder_rules():
    vaswani = IndexPaperDetail(
        key="1706.03762",
        title="Attention Is All You Need",
        year=2017,
        authors=["Ashish Vaswani", "Noam Shazeer"],
        venue="NeurIPS",
        arxiv_id="1706.03762",
    )
    assert cite_key(vaswani) == "vaswani2017attention"
    b = bibtex(vaswani)
    assert b.startswith("@inproceedings{vaswani2017attention,")
    assert "booktitle = {NeurIPS}" in b and "eprint = {1706.03762}" in b
    j = bibtex(
        IndexPaperDetail(
            key="x",
            title="Q & A: 50% of $x_1$",
            year=2020,
            venue="Journal of ML Research",
            authors=["Ana Ruiz"],
        )
    )
    assert j.startswith("@article{") and r"Q \& A: 50\% of \$x\_1\$" in j
    anon = bibtex(IndexPaperDetail(key="openalex:W1", title="Old Work", year=1998, doi="10.1/x"))
    assert anon.startswith("% authors not in the Reviewer Zero index")
    assert "url = {https://doi.org/10.1/x}" in anon
    twins = [
        IndexPaperDetail(key=str(i), title="Deep Nets", year=2020, authors=["X Lee"])
        for i in range(3)
    ]
    assert unique_keys(twins) == ["lee2020deep" + s for s in "abc"]
    many = [twins[0]] * 28
    assert unique_keys(many)[25:] == ["lee2020deepz", "lee2020deepaa", "lee2020deepab"]
    for text in (b, j, anon):
        assert len(parse_string(text, "bibtex").entries) == 1


@pytest.mark.parametrize(
    ("queries", "k", "message"),
    [
        ([], 40, "6 to 10 component queries"),
        (["too short"], 40, "Query 1 has 2 words"),
        (["word " * 61], 40, "Query 1 has 61 words"),
        (QUERIES, 41, "k must be between 1 and 40"),
        (QUERIES * 2, 40, "got 14"),
    ],
)
def test_validation_messages(queries, k, message):
    with pytest.raises(QueryError, match=message):
        validate(queries, k)


def test_tool_through_an_mcp_client(index):
    async def go():
        async with Client(build_server(index_factory=lambda: index)) as c:
            tools = {t.name: t for t in (await c.list_tools()).tools}
            ok = await c.call_tool(
                "find_prior_work", {"queries": QUERIES, "before": "2025-06-01", "k": 5}
            )
            bad = await c.call_tool("find_prior_work", {"queries": ["two words"]})
            return tools, ok, bad

    tools, ok, bad = anyio.run(go)
    desc = tools["find_prior_work"].description
    assert "6 to 10 component queries" in desc and "No telemetry" in desc
    assert not ok.is_error and len(ok.structured_content["candidates"]) == 5
    assert bad.is_error and "words" in bad.content[0].text


def test_searches_and_expand_batches_run_concurrently(tmp_path):
    import threading
    import time as _time

    inner = FakeIndexServer()
    lock = threading.Lock()
    state = {"now": {"search": 0, "expand": 0}, "peak": {"search": 0, "expand": 0}}

    def slow(req: httpx.Request) -> httpx.Response:
        kind = req.url.path.strip("/")
        if kind in state["now"]:
            with lock:
                state["now"][kind] += 1
                state["peak"][kind] = max(state["peak"][kind], state["now"][kind])
            _time.sleep(0.05)
            with lock:
                state["now"][kind] -= 1
        return inner(req)

    idx = HostedIndex(
        McpConfig(index_url="https://index.test", index_key="k", cache=tmp_path),
        transport=httpx.MockTransport(slow),
        sleep=lambda s: None,
    )
    out = run_find_prior_work(idx, QUERIES, BEFORE, k=40)
    assert len(out.candidates) == 40
    assert state["peak"]["search"] > 1  # 7 searches overlap
    assert state["peak"]["expand"] == 2  # both /expand batches in flight together
