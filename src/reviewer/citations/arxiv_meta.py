"""arXiv records for cited preprints: abstract, authors, v1 date, and the author-supplied
journal_ref / DOI (evidence that a preprint was later published at a venue).

Uses the same curl transport as `reviewer.corpus.arxiv_abstracts` (export.arxiv.org rejects
Python TLS clients on cache misses), with its own cache namespace because it keeps more fields.
"""

import logging
import re
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from urllib.parse import urlencode

from lxml import etree
from pydantic import BaseModel

from reviewer.cache import DiskCache, content_key
from reviewer.config import Settings
from reviewer.corpus.arxiv_abstracts import (
    _HTTP_CODE_MARKER,
    BATCH_SIZE,
    CURL_MAX_TIME_S,
    RETRY_ATTEMPTS,
    RETRY_BACKOFF_S,
    RETRYABLE_STATUSES,
    Runner,
    curl_args,
    query_url,
)
from reviewer.corpus.arxiv_client import API_URL
from reviewer.corpus.ids import strip_version

logger = logging.getLogger(__name__)

CACHE_VERSION = "arxiv-meta-v2"
SEARCH_CACHE_VERSION = "arxiv-title-search-v2"
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "via",
        "with",
    ]
)
_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


class ArxivRecord(BaseModel):
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    published: datetime
    """v1 submission date."""
    updated: datetime | None = None
    """Date of the latest version."""
    journal_ref: str | None = None
    doi: str | None = None


def _text(el: etree._Element, path: str) -> str | None:
    found = el.find(path, _NS)
    if found is None or found.text is None:
        return None
    return " ".join(found.text.split()) or None


def parse_records(xml: bytes) -> dict[str, ArxivRecord]:
    root = etree.fromstring(xml)
    out: dict[str, ArxivRecord] = {}
    for entry in root.findall("atom:entry", _NS):
        url = _text(entry, "atom:id")
        title = _text(entry, "atom:title")
        published = _text(entry, "atom:published")
        updated = _text(entry, "atom:updated")
        if not url or not title or not published:
            continue
        arxiv_id = strip_version(url.split("/abs/")[-1])
        out[arxiv_id] = ArxivRecord(
            arxiv_id=arxiv_id,
            title=title,
            abstract=_text(entry, "atom:summary") or "",
            authors=[
                name
                for a in entry.findall("atom:author", _NS)
                if (name := _text(a, "atom:name")) is not None
            ],
            published=datetime.fromisoformat(published.replace("Z", "+00:00")),
            updated=datetime.fromisoformat(updated.replace("Z", "+00:00")) if updated else None,
            journal_ref=_text(entry, "arxiv:journal_ref"),
            doi=_text(entry, "arxiv:doi"),
        )
    return out


class ArxivMetaClient:
    """Batched, cached lookups, `arxiv_delay_s` apart. Missing ids are simply absent."""

    def __init__(
        self,
        settings: Settings,
        runner: Runner = subprocess.run,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self._run = runner
        self._sleep = sleep
        self._made_request = False
        self._lock = threading.Lock()
        self.cache = DiskCache(settings.cache_dir, "arxiv_meta", ".json")

    def _fetch_once(self, url: str) -> tuple[dict[str, ArxivRecord] | None, str]:
        with self._lock:  # arXiv asks for one request at a time, `arxiv_delay_s` apart
            if self._made_request:
                self._sleep(self.settings.arxiv_delay_s)
            self._made_request = True
            return self._curl(url)

    def _curl(self, url: str) -> tuple[dict[str, ArxivRecord] | None, str]:
        try:
            proc = self._run(
                curl_args(url, self.settings.user_agent),
                capture_output=True,
                check=False,
                timeout=CURL_MAX_TIME_S + 15,
            )
        except (OSError, subprocess.SubprocessError) as e:
            logger.warning("curl failed for %s: %s", url, e)
            return None, "curl-error"
        body, sep, code = (proc.stdout or b"").rpartition(_HTTP_CODE_MARKER.encode())
        status = code.decode(errors="replace").strip() if sep else "?"
        if proc.returncode != 0 or status != "200":
            return None, status
        try:
            return parse_records(body), status
        except etree.XMLSyntaxError:
            return None, "bad-xml"

    def _fetch(self, url: str) -> dict[str, ArxivRecord] | None:
        """Parsed records, or None if the request failed after retries."""
        for attempt in range(RETRY_ATTEMPTS):
            result, status = self._fetch_once(url)
            if result is not None:
                return result
            if status not in RETRYABLE_STATUSES or attempt == RETRY_ATTEMPTS - 1:
                logger.warning("arXiv API request failed (HTTP %s): %s", status, url)
                break
            self._sleep(RETRY_BACKOFF_S * (2**attempt))
        return None

    def lookup(self, ids: Sequence[str]) -> dict[str, ArxivRecord]:
        wanted = list(dict.fromkeys(strip_version(i) for i in ids))
        out: dict[str, ArxivRecord] = {}
        todo: list[str] = []
        for arxiv_id in wanted:
            cached = self.cache.get_json(content_key(CACHE_VERSION, arxiv_id))
            if cached is not None:
                out[arxiv_id] = ArxivRecord.model_validate(cached)
            else:
                todo.append(arxiv_id)
        for start in range(0, len(todo), BATCH_SIZE):
            batch = todo[start : start + BATCH_SIZE]
            fetched = self._fetch(query_url(batch)) or {}
            for arxiv_id, record in fetched.items():
                self.cache.put_json(content_key(CACHE_VERSION, arxiv_id), record.model_dump())
                out[arxiv_id] = record
        return out

    def search_title(self, title: str, max_results: int = 5) -> list[ArxivRecord] | None:
        """arXiv title search (all words of `title`). None if the request failed."""
        kept = [w for w in title.split() if "'" not in w and "’" not in w]
        words = [
            w
            for w in re.sub(r"[^\w\s]", " ", " ".join(kept)).split()
            if len(w) > 1 and w.lower() not in STOPWORDS
        ]
        if not words:
            return []
        key = content_key(SEARCH_CACHE_VERSION, " ".join(words).lower())
        cached = self.cache.get_json(key)
        if cached is not None:
            return [ArxivRecord.model_validate(r) for r in cached["records"]]
        query = " AND ".join(f"ti:{w}" for w in words[:12])
        url = f"{API_URL}?" + urlencode({"search_query": query, "max_results": str(max_results)})
        fetched = self._fetch(url)
        if fetched is None:
            return None
        records = list(fetched.values())
        self.cache.put_json(key, {"records": [r.model_dump(mode="json") for r in records]})
        return records
