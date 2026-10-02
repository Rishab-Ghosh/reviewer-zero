"""Thin cached client for the reviewer-index query API.

Contract (agreed 2026-09-25; JSON bodies, unknown fields ignored):

- `GET /health` -> `{"index_version": str, ...}`
- `GET /search?q=<text>&k=<int>[&before=YYYY-MM-DD]` -> `[paper, ...]`, most similar first
- `GET /neighbors?key=<paper key>&q=<text>[&before=YYYY-MM-DD]` -> `[paper, ...]` (one hop,
  both directions; 404 when the key is unknown)

`before` is applied inside the index, so k results come back from before the cutoff.

- `POST /expand` `{seeds, queries (<=20), before?, model?, max_candidates}` ->
  `{candidates: [{key, kind, title, abstract, year, seed_count, seeds: [{seed, relation}],
  max_cosine}], total}`: every one-hop citation neighbor of any seed, `before` applied.

where `paper` = `{"key", "kind", "title", "abstract", "published_date" (YYYY-MM-DD),
"year", "similarity" (to q), "relation" ("cites" | "cited_by", neighbors only)}`. `year` must
be present (null allowed); title/abstract are null for known-bad metadata.
A body wrapped as `{"results": [...]}` is accepted too.

Responses are cached on disk under a key that includes the index version, so a daily index
update invalidates them.
"""

import json
import logging
from datetime import date
from typing import Protocol

import httpx
from pydantic import BaseModel, ValidationError

from reviewer.cache import DiskCache, content_key
from reviewer.config import Settings
from reviewer.models import (
    ExpandResponse,
    IndexHealth,
    IndexNeighborsResponse,
    IndexPaper,
    IndexSearchResponse,
)

logger = logging.getLogger(__name__)

CACHE_VERSION = "index-v1"


class IndexClientError(RuntimeError):
    """reviewer-index unreachable, or it answered with an error or an unreadable body."""


class IndexAPI(Protocol):
    def version(self) -> str: ...

    def search(self, q: str, k: int, before: date | None = None) -> list[IndexPaper]: ...

    def neighbors(self, key: str, q: str, before: date | None = None) -> list[IndexPaper]: ...

    def expand(
        self, seeds: list[str], queries: list[str], before: date | None, max_candidates: int
    ) -> ExpandResponse: ...


class HttpIndexClient:
    def __init__(self, settings: Settings, http: httpx.Client | None = None) -> None:
        self.base_url = settings.index_url.rstrip("/")
        self.http = http or httpx.Client(timeout=settings.index_timeout_s)
        self.cache = DiskCache(settings.cache_dir, "index", ".json")
        self.replay = settings.index_replay_version is not None
        self.embed_model = settings.novelty_embed_model
        self._version: str | None = settings.index_replay_version

    def _get(self, path: str, params: dict[str, str | int]) -> httpx.Response:
        try:
            return self.http.get(f"{self.base_url}{path}", params=params)
        except httpx.HTTPError as e:
            raise IndexClientError(f"GET {path}: {type(e).__name__}: {e}") from e

    def version(self) -> str:
        if self._version is None:
            r = self._get("/health", {})
            if r.status_code != 200:
                raise IndexClientError(f"GET /health: HTTP {r.status_code}: {r.text[:200]}")
            self._version = _validate(IndexHealth, r, "/health").index_version
            logger.info("reviewer-index at %s, version %s", self.base_url, self._version)
            if self.embed_model:
                self._check_model_param()
        return self._version

    def _check_model_param(self) -> None:
        """A server that predates `model` silently ignores it and serves the default space.
        It must reject an unknown model; otherwise results would be mislabelled."""
        r = self._get("/search", {"q": "probe", "k": 1, "model": "__no_such_model__"})
        if r.status_code == 200:
            raise IndexClientError(
                f"reviewer-index at {self.base_url} ignores the model parameter (an unknown model "
                f"was accepted), so model={self.embed_model} would silently use the default; "
                "restart `reviewer-index serve` on a version that supports it"
            )

    def _cached[T: BaseModel](
        self, path: str, params: dict[str, str | int], schema: type[T], not_found_ok: bool
    ) -> T | None:
        key = content_key(CACHE_VERSION, self.version(), path, *_flat(params))
        if (hit := self.cache.get_json(key)) is not None:
            return None if hit.get("not_found") else schema.model_validate(hit)
        if self.replay:
            raise IndexClientError(
                f"replay of index {self._version}: no cached response for {path} {params}"
            )
        r = self._get(path, params)
        if r.status_code == 404 and not_found_ok:
            self.cache.put_json(key, {"not_found": True})
            return None
        if r.status_code != 200:
            raise IndexClientError(f"GET {path} {params}: HTTP {r.status_code}: {r.text[:200]}")
        parsed = _validate(schema, r, path)
        self.cache.put_json(key, parsed.model_dump(mode="json"))
        return parsed

    def search(self, q: str, k: int, before: date | None = None) -> list[IndexPaper]:
        params = _with_before({"q": q, "k": k}, before)
        if self.embed_model:
            params["model"] = self.embed_model
        resp = self._cached("/search", params, IndexSearchResponse, not_found_ok=False)
        assert resp is not None
        return resp.results

    def neighbors(self, key: str, q: str, before: date | None = None) -> list[IndexPaper]:
        params = _with_before({"key": key, "q": q}, before)
        resp = self._cached("/neighbors", params, IndexNeighborsResponse, not_found_ok=True)
        if resp is None:
            logger.info("neighbors: %s not in index", key)
            return []
        return resp.results

    def expand(
        self, seeds: list[str], queries: list[str], before: date | None, max_candidates: int
    ) -> ExpandResponse:
        """POST /expand. `target` is never sent: excluding what the paper cites would use its
        reference list."""
        body: dict[str, object] = {
            "seeds": seeds,
            "queries": queries,
            "max_candidates": max_candidates,
        }
        if before is not None:
            body["before"] = before.isoformat()
        if self.embed_model:
            body["model"] = self.embed_model
        key = content_key(
            CACHE_VERSION, self.version(), "POST /expand", json.dumps(body, sort_keys=True)
        )
        if (hit := self.cache.get_json(key)) is not None:
            return ExpandResponse.model_validate(hit)
        if self.replay:
            raise IndexClientError(f"replay of index {self._version}: no cached /expand response")
        try:
            r = self.http.post(f"{self.base_url}/expand", json=body)
        except httpx.HTTPError as e:
            raise IndexClientError(f"POST /expand: {type(e).__name__}: {e}") from e
        if r.status_code != 200:
            raise IndexClientError(f"POST /expand: HTTP {r.status_code}: {r.text[:200]}")
        try:
            resp = ExpandResponse.model_validate_json(r.content)
        except ValidationError as e:
            raise IndexClientError(f"POST /expand: unexpected response body: {e}") from e
        self.cache.put_json(key, resp.model_dump(mode="json"))
        return resp


def _with_before(params: dict[str, str | int], before: date | None) -> dict[str, str | int]:
    """The index applies `before` inside its search (only papers older than the date)."""
    return params if before is None else {**params, "before": before.isoformat()}


def _flat(params: dict[str, str | int]) -> list[str]:
    return [f"{k}={params[k]}" for k in sorted(params)]


def _require_year(body: object, path: str) -> None:
    """Every paper must carry a `year` key (its value may be null): without it the cutoff
    filter would drop every out-of-corpus work, and that answer would be cached."""
    results = body.get("results") if isinstance(body, dict) else None
    if isinstance(results, list) and any(isinstance(p, dict) and "year" not in p for p in results):
        raise IndexClientError(
            f"GET {path}: results have no 'year' field; this reviewer-index predates the "
            "contract's year field"
        )


def _validate[T: BaseModel](schema: type[T], r: httpx.Response, path: str) -> T:
    try:
        body = r.json()
        if isinstance(body, list):
            body = {"results": body}
        _require_year(body, path)
        return schema.model_validate(body)
    except (ValueError, ValidationError) as e:
        raise IndexClientError(f"GET {path}: unexpected response body: {e}") from e
