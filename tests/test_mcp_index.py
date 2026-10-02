import json

import httpx
import pytest

from reviewer.mcp.config import McpConfig
from reviewer.mcp.index import (
    HostedIndex,
    IndexAuthError,
    IndexNotConfiguredError,
    IndexRateLimitedError,
    retry_after_seconds,
)
from reviewer.novelty.index_client import IndexClientError

KEY = "rz_test_key_1234567890"
HEALTH = {"index_version": "v1"}
HIT = {"key": "2106.09685", "kind": "arxiv", "title": "LoRA", "year": 2021, "similarity": 0.8}


def _config(tmp_path, **kw) -> McpConfig:
    return McpConfig(index_url="https://index.test", index_key=KEY, cache=tmp_path / "cache", **kw)


def _index(tmp_path, handler, sleeps=None, **kw) -> HostedIndex:
    return HostedIndex(
        _config(tmp_path, **kw),
        transport=httpx.MockTransport(handler),
        sleep=(sleeps.append if sleeps is not None else lambda s: None),
    )


def test_every_request_carries_the_key_header(tmp_path):
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        if req.url.path == "/health":
            return httpx.Response(200, json=HEALTH)
        if req.url.path == "/search":
            return httpx.Response(200, json=[HIT])
        return httpx.Response(200, json={"candidates": [], "total": 0})

    idx = _index(tmp_path, handler)
    assert [p.key for p in idx.search("low-rank adaptation", 5)] == ["2106.09685"]
    idx.expand(["2106.09685"], ["low-rank adaptation"], None, 100)
    assert len(seen) == 3
    assert all(r.headers.get("X-API-Key") == KEY for r in seen)


def test_no_key_sends_no_header(tmp_path):
    seen: list[httpx.Request] = []

    def handler(req):
        seen.append(req)
        return httpx.Response(200, json=HEALTH)

    HostedIndex(
        McpConfig(index_url="https://index.test", cache=tmp_path),
        transport=httpx.MockTransport(handler),
    ).version()
    assert "X-API-Key" not in seen[0].headers


def test_401_raises_auth_error_without_leaking_the_key(tmp_path):
    idx = _index(tmp_path, lambda req: httpx.Response(401, json={"error": "invalid_api_key"}))
    with pytest.raises(IndexAuthError) as e:
        idx.version()
    assert "REVIEWER_ZERO_INDEX_KEY" in str(e.value)
    assert KEY not in str(e.value)


def test_429_with_long_retry_after_raises_with_the_wait(tmp_path):
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(429, headers={"Retry-After": "120"})

    sleeps: list[float] = []
    idx = _index(tmp_path, handler, sleeps)
    with pytest.raises(IndexRateLimitedError) as e:
        idx.version()
    assert e.value.retry_after_s == 120.0
    assert "about 120 s" in str(e.value) and KEY not in str(e.value)
    assert len(calls) == 1 and sleeps == []  # too long to wait inside a tool call


def test_429_with_short_retry_after_waits_then_succeeds(tmp_path):
    responses = [
        httpx.Response(429, headers={"Retry-After": "2"}),
        httpx.Response(200, json=HEALTH),
    ]
    sleeps: list[float] = []
    idx = _index(tmp_path, lambda req: responses.pop(0), sleeps)
    assert idx.version() == "v1"
    assert sleeps == [2.0]


def test_503_is_retried_with_backoff_then_gives_up(tmp_path):
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(503)

    sleeps: list[float] = []
    idx = _index(tmp_path, handler, sleeps, index_retries=2)
    with pytest.raises(IndexClientError, match="503"):
        idx.version()
    assert len(calls) == 3 and sleeps == [0.5, 1.0]


def test_connection_error_is_retried(tmp_path):
    attempts = []

    def handler(req):
        attempts.append(req)
        if len(attempts) == 1:
            raise httpx.ConnectError("refused", request=req)
        return httpx.Response(200, json=HEALTH)

    assert _index(tmp_path, handler).version() == "v1"
    assert len(attempts) == 2


def test_retry_after_parsing():
    assert retry_after_seconds("7") == 7.0
    assert retry_after_seconds(None) is None
    assert retry_after_seconds("soon") is None
    assert retry_after_seconds("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0  # in the past


def test_papers_uses_batch_route(tmp_path):
    def handler(req):
        assert req.url.path == "/papers" and json.loads(req.content) == {"keys": ["a", "b"]}
        return httpx.Response(
            200, json={"papers": [{"key": "a", "title": "A"}, {"key": "b"}], "missing": []}
        )

    out = _index(tmp_path, handler).papers(["a", "b"])
    assert set(out) == {"a", "b"} and out["a"].title == "A"


def test_papers_falls_back_to_single_lookups_on_an_older_index(tmp_path):
    def handler(req):
        if req.url.path == "/papers":
            return httpx.Response(404, json={"detail": "Not Found"})  # route missing
        if req.url.path == "/paper/a":
            return httpx.Response(200, json={"key": "a", "authors": ["X Y"], "year": 2020})
        return httpx.Response(404, json={"error": "not_found", "detail": "Paper not found."})

    out = _index(tmp_path, handler).papers(["a", "b"])
    assert list(out) == ["a"] and out["a"].authors == ["X Y"]


def test_citations_route_and_neighbors_fallback(tmp_path):
    n = [dict(HIT, relation="cites"), dict(HIT, key="x", relation="cited_by")]

    def new_index(req):
        assert req.url.path == "/citations/2106.09685"
        assert req.url.params["direction"] == "cited_by"
        return httpx.Response(200, json=[n[1]])

    def old_index(req):
        if req.url.path.startswith("/citations/"):  # public-mode unknown-route envelope
            return httpx.Response(404, json={"error": "not_found", "detail": "Not Found"})
        return httpx.Response(200, json={"results": n})

    assert [p.key for p in _index(tmp_path, new_index).citations("2106.09685", "cited_by")] == ["x"]
    got = _index(tmp_path, old_index).citations("2106.09685", "cites")
    assert [p.key for p in got] == ["2106.09685"]


def test_unknown_key_returns_none(tmp_path):
    def handler(req):
        return httpx.Response(404, json={"error": "not_found", "detail": "Paper not found."})

    idx = _index(tmp_path, handler)
    assert idx.paper("zz") is None
    assert idx.citations("zz") is None


def test_hosted_index_is_the_default_and_an_empty_url_is_a_clear_error(tmp_path):
    assert McpConfig(cache=tmp_path).index_url == "https://api.tryreviewerzero.com"
    with pytest.raises(IndexNotConfiguredError, match="REVIEWER_ZERO_INDEX_URL"):
        HostedIndex(McpConfig(cache=tmp_path, index_url=""))


def test_config_reads_only_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("REVIEWER_ZERO_INDEX_URL", "https://example.test")
    monkeypatch.setenv("REVIEWER_ZERO_INDEX_KEY", KEY)
    monkeypatch.setenv("REVIEWER_ZERO_CACHE", str(tmp_path))
    c = McpConfig()
    assert c.index_url == "https://example.test" and c.cache == tmp_path
    assert KEY not in repr(c)
    s = c.pipeline_settings()
    assert s.index_url == "https://example.test"
    assert s.cache_dir == tmp_path / "public" and s.data_dir == tmp_path / "work"


def test_long_retry_after_is_waited_out_when_configured(tmp_path):
    responses = [
        httpx.Response(429, headers={"Retry-After": "20"}),
        httpx.Response(200, json=HEALTH),
    ]
    sleeps: list[float] = []
    idx = _index(tmp_path, lambda req: responses.pop(0), sleeps, index_max_retry_after_s=30.0)
    assert idx.version() == "v1" and sleeps == [20.0]
