"""Cached JSON GETs for the citation sources (reviewer-index, OpenAlex, Crossref).

Responses are cached by (namespace, URL, sorted params minus secrets). 200 and 404 are cached
so re-runs are deterministic; 429/5xx/transport errors are retried, then reported, never cached.
"""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from reviewer.cache import DiskCache, content_key

logger = logging.getLogger(__name__)

SECRET_PARAMS = frozenset({"api_key", "mailto"})
RETRYABLE = frozenset({429, 500, 502, 503, 504})
CACHE_VERSION = "citations-http-v1"


class SourceError(Exception):
    """A lookup that failed after retries (network error, 429, 5xx, bad JSON)."""


@dataclass(frozen=True)
class JsonResponse:
    status: int
    body: Any
    """Parsed JSON (None for 404)."""
    headers: dict[str, str]
    """Only for fresh responses; empty when served from cache."""
    from_cache: bool


class CachedJsonClient:
    def __init__(
        self,
        cache: DiskCache,
        *,
        user_agent: str,
        timeout_s: float = 30.0,
        attempts: int = 3,
        backoff_s: float = 2.0,
        min_interval_s: float = 0.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.cache = cache
        self.attempts = attempts
        self.backoff_s = backoff_s
        self.min_interval_s = min_interval_s
        self._sleep = sleep
        self._lock = threading.Lock()
        self._last = 0.0
        self._http = httpx.Client(
            transport=transport,
            headers={"User-Agent": user_agent, **(headers or {})},
            follow_redirects=True,
            timeout=httpx.Timeout(timeout_s, connect=5.0),
        )

    def close(self) -> None:
        self._http.close()

    @staticmethod
    def key(url: str, params: dict[str, str]) -> str:
        public = sorted((k, v) for k, v in params.items() if k not in SECRET_PARAMS)
        return content_key(CACHE_VERSION, url, repr(public))

    def _throttle(self) -> None:
        if self.min_interval_s <= 0:
            return
        with self._lock:
            wait = self.min_interval_s - (time.monotonic() - self._last)
            if wait > 0:
                self._sleep(wait)
            self._last = time.monotonic()

    def cached(self, url: str, params: dict[str, str] | None = None) -> JsonResponse | None:
        hit = self.cache.get_json(self.key(url, params or {}))
        if hit is None:
            return None
        return JsonResponse(hit["status"], hit["body"], {}, from_cache=True)

    def get(
        self, url: str, params: dict[str, str] | None = None, *, use_cache: bool = True
    ) -> JsonResponse:
        params = params or {}
        key = self.key(url, params)
        if use_cache and (hit := self.cached(url, params)) is not None:
            return hit
        last_error = "no attempt made"
        for attempt in range(self.attempts):
            self._throttle()
            try:
                resp = self._http.get(url, params=params)
            except httpx.HTTPError as e:
                last_error = f"{type(e).__name__}: {e}"
            else:
                if resp.status_code in (200, 404):
                    body = None
                    if resp.status_code == 200:
                        try:
                            body = resp.json()
                        except ValueError as e:
                            raise SourceError(f"bad JSON from {url}: {e}") from e
                    if use_cache:
                        self.cache.put_json(key, {"status": resp.status_code, "body": body})
                    return JsonResponse(resp.status_code, body, dict(resp.headers), False)
                last_error = f"HTTP {resp.status_code}"
                if resp.status_code not in RETRYABLE:
                    break
                # OpenAlex answers 429 when the daily budget is gone; retrying won't help.
                budget_gone = "x-ratelimit-remaining-usd" in resp.headers
                if resp.status_code == 429 and "retry-after" not in resp.headers and budget_gone:
                    break
            if attempt < self.attempts - 1:
                self._sleep(self.backoff_s * (2**attempt))
        raise SourceError(f"GET {url} failed: {last_error}")
