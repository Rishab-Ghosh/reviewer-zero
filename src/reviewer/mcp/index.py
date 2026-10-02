"""The hosted index for MCP tools: the pipeline's cached HTTP client plus the API-key header,
retries, typed auth/quota errors and the metadata lookups (MCP_PLAN §8).

Implements the pipeline's `IndexAPI` protocol, so `novelty/retrieve.py` and
`novelty/firststage.py` run against it unchanged.
"""

import contextlib
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from email.utils import parsedate_to_datetime
from typing import Literal

import httpx

from reviewer.index_gate import MAX_CONCURRENT_INDEX_REQUESTS, IndexGateTransport
from reviewer.mcp.config import McpConfig
from reviewer.models import (
    ExpandCandidate,
    ExpandResponse,
    ExpandSeedLink,
    IndexPaper,
    IndexPaperDetail,
)
from reviewer.novelty.index_client import HttpIndexClient, IndexClientError

logger = logging.getLogger(__name__)

KEY_HINT = (
    "Set REVIEWER_ZERO_INDEX_KEY in the environment that starts the MCP server. "
    "Keys are free on request (see the README)."
)
MAX_RETRY_AFTER_WAIT_S = 5.0
RETRY_STATUSES = frozenset({502, 503, 504})
PUBLIC_EXPAND_MAX_QUERIES = 5
PUBLIC_EXPAND_MAX_CANDIDATES = 500
MAX_PARALLEL = MAX_CONCURRENT_INDEX_REQUESTS
"""Worker threads per tool call; the shared gate (index_gate) enforces the per-key limit."""


class IndexNotConfiguredError(IndexClientError):
    """No index URL."""


class IndexAuthError(IndexClientError):
    """401/403: missing, unknown or revoked API key."""


class IndexRateLimitedError(IndexClientError):
    """429 with a Retry-After too long to wait inside a tool call."""

    def __init__(self, message: str, retry_after_s: float | None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


def retry_after_seconds(value: str | None) -> float | None:
    """Retry-After as delta-seconds or an HTTP date; None when absent or unreadable."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
    except (TypeError, ValueError):
        return None


class RetryingTransport(httpx.BaseTransport):
    """Retries connection errors and 502/503/504 with exponential backoff, and a 429 whose
    Retry-After is at most MAX_RETRY_AFTER_WAIT_S; anything else is returned as is."""

    def __init__(
        self,
        inner: httpx.BaseTransport,
        retries: int,
        sleep: Callable[[float], None] = time.sleep,
        max_retry_after_s: float = MAX_RETRY_AFTER_WAIT_S,
    ) -> None:
        self.inner = inner
        self.retries = retries
        self.sleep = sleep
        self.max_retry_after_s = max_retry_after_s

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        for attempt in range(self.retries + 1):
            last = attempt == self.retries
            try:
                response = self.inner.handle_request(request)
            except httpx.TransportError:
                if last:
                    raise
                self.sleep(0.5 * 2**attempt)
                continue
            if response.status_code in RETRY_STATUSES and not last:
                response.close()
                self.sleep(0.5 * 2**attempt)
                continue
            if response.status_code == 429 and not last:
                wait = retry_after_seconds(response.headers.get("retry-after"))
                if wait is not None and wait <= self.max_retry_after_s:
                    response.close()
                    self.sleep(wait)
                    continue
            return response
        raise AssertionError("unreachable")

    def close(self) -> None:
        self.inner.close()


def _raise_for_auth_and_quota(response: httpx.Response) -> None:
    path = response.request.url.path
    if response.status_code in (401, 403):
        raise IndexAuthError(f"The Reviewer Zero index refused the API key ({path}). {KEY_HINT}")
    if response.status_code == 429:
        wait = retry_after_seconds(response.headers.get("retry-after"))
        when = f" Try again in about {round(wait)} s." if wait is not None else ""
        raise IndexRateLimitedError(
            f"The Reviewer Zero index's quota for this API key is used up for now.{when}", wait
        )


class HostedIndex(HttpIndexClient):
    def __init__(
        self,
        config: McpConfig,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not config.index_url:
            raise IndexNotConfiguredError(
                "No index URL: set REVIEWER_ZERO_INDEX_URL in the environment that starts the "
                "MCP server."
            )
        headers = {"X-API-Key": config.index_key} if config.index_key else {}
        http = httpx.Client(
            headers=headers,
            timeout=config.index_timeout_s,
            transport=RetryingTransport(
                IndexGateTransport(transport),
                config.index_retries,
                sleep,
                config.index_max_retry_after_s,
            ),
            event_hooks={"response": [_raise_for_auth_and_quota]},
        )
        super().__init__(config.pipeline_settings(), http=http)

    def prefetch_searches(self, queries: list[str], k: int, before: date | None) -> None:
        """Run every /search concurrently so the pipeline's sequential `run_searches` then reads
        them from the disk cache. Errors are left for the sequential pass to raise."""
        self.version()  # once, before the threads (it is part of every cache key)
        with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL, len(queries))) as pool:
            for f in [pool.submit(self.search, q, k, before) for q in queries]:
                with contextlib.suppress(IndexClientError):
                    f.result()

    def expand(
        self, seeds: list[str], queries: list[str], before: date | None, max_candidates: int
    ) -> ExpandResponse:
        """The public /expand takes ≤ 5 queries and ≤ 500 candidates per call: send the
        queries in batches, concurrently, and merge (seed counts are per candidate, cosines
        take the max)."""
        cap = min(max_candidates, PUBLIC_EXPAND_MAX_CANDIDATES)
        batches = [
            queries[i : i + PUBLIC_EXPAND_MAX_QUERIES]
            for i in range(0, len(queries), PUBLIC_EXPAND_MAX_QUERIES)
        ]
        self.version()
        with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL, len(batches) or 1)) as pool:
            responses = list(
                pool.map(lambda b: HttpIndexClient.expand(self, seeds, b, before, cap), batches)
            )
        merged: dict[str, ExpandCandidate] = {}
        total = 0
        for resp in responses:
            total = max(total, resp.total)
            for c in resp.candidates:
                if (old := merged.get(c.key)) is None:
                    merged[c.key] = c
                    continue
                links = {(s.seed, s.relation): s for s in [*old.seeds, *c.seeds]}
                cosines = [x for x in (old.max_cosine, c.max_cosine) if x is not None]
                merged[c.key] = old.model_copy(
                    update={
                        "seed_count": max(old.seed_count, c.seed_count),
                        "seeds": [ExpandSeedLink.model_validate(s) for s in links.values()],
                        "max_cosine": max(cosines) if cosines else None,
                    }
                )
        return ExpandResponse(candidates=list(merged.values()), total=total)

    # ------------------------------------------------------------------ metadata lookups

    def paper(self, key: str) -> IndexPaperDetail | None:
        r = self._get(f"/paper/{key}", {})
        if r.status_code == 404:
            return None
        if r.status_code != 200:
            raise IndexClientError(f"GET /paper: HTTP {r.status_code}")
        return IndexPaperDetail.model_validate(r.json())

    def papers(self, keys: list[str]) -> dict[str, IndexPaperDetail]:
        """Batch metadata (POST /papers, ≤ 50 keys per call, response `{papers, missing}`);
        falls back to one GET /paper per key on an index without the batch route."""
        out: dict[str, IndexPaperDetail] = {}
        for i in range(0, len(keys), 50):
            chunk = keys[i : i + 50]
            try:
                r = self.http.post(f"{self.base_url}/papers", json={"keys": chunk})
            except httpx.HTTPError as e:
                raise IndexClientError(f"POST /papers: {type(e).__name__}") from e
            if _route_missing(r):
                for key in chunk:
                    if (p := self.paper(key)) is not None:
                        out[p.key] = p
                continue
            if r.status_code != 200:
                raise IndexClientError(f"POST /papers: HTTP {r.status_code}")
            for row in _rows(r):
                p = IndexPaperDetail.model_validate(row)
                out[p.key] = p
        return out

    def citations(
        self, key: str, direction: Literal["cites", "cited_by"] = "cites", limit: int = 50
    ) -> list[IndexPaper] | None:
        """One-hop neighbours of `key` in one direction; None if the key is unknown. Uses
        GET /citations/{key} when the index has it, else GET /neighbors filtered by relation."""
        r = self._get(f"/citations/{key}", {"direction": direction, "limit": limit})
        if _route_missing(r):
            r = self._get("/neighbors", {"key": key})
            if r.status_code == 404:
                return None
            if r.status_code != 200:
                raise IndexClientError(f"GET /neighbors: HTTP {r.status_code}")
            papers = [IndexPaper.model_validate(row) for row in _rows(r)]
            return [p for p in papers if p.relation == direction][:limit]
        if r.status_code == 404:
            return None
        if r.status_code != 200:
            raise IndexClientError(f"GET /citations: HTTP {r.status_code}")
        return [IndexPaper.model_validate(row) for row in _rows(r)][:limit]


def _route_missing(r: httpx.Response) -> bool:
    """True when the index predates a route: 405, or a 404 whose detail is the framework's
    "Not Found" for an unknown path (plain FastAPI `{"detail": "Not Found"}`, public mode
    `{"error": "not_found", "detail": "Not Found"}`), as opposed to "Paper not found."."""
    if r.status_code == 405:
        return True
    if r.status_code != 404:
        return False
    try:
        body = r.json()
    except ValueError:
        return False
    return isinstance(body, dict) and body.get("detail") == "Not Found"


def _rows(r: httpx.Response) -> list[dict]:
    try:
        body = r.json()
    except ValueError as e:
        raise IndexClientError(f"{r.request.url.path}: response is not JSON") from e
    if isinstance(body, dict):
        body = body.get("results", body.get("papers", body))
    rows = body
    if not isinstance(rows, list):
        raise IndexClientError(f"{r.request.url.path}: unexpected response body")
    return rows
