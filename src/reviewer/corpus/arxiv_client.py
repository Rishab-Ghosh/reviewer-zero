"""Thin arXiv client: polite PDF downloads and Atom-API metadata lookups."""

import logging
import time
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path

import httpx
from lxml import etree
from pydantic import BaseModel

from reviewer.config import Settings
from reviewer.corpus.ids import strip_version

logger = logging.getLogger(__name__)

PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"
API_URL = "https://export.arxiv.org/api/query"
PDF_MAGIC = b"%PDF"
METADATA_BATCH = 50

_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


class ArxivError(Exception):
    """A download or lookup that cannot succeed (404, non-PDF body, HTTP error)."""


class ArxivMeta(BaseModel):
    arxiv_id: str
    """Version-less arXiv id."""
    title: str
    published: datetime
    """Submission date of v1."""
    primary_category: str
    categories: list[str]
    comment: str | None = None


def _text(el: etree._Element, path: str) -> str | None:
    found = el.find(path, _NS)
    if found is None or found.text is None:
        return None
    return " ".join(found.text.split())


def parse_atom_feed(xml: bytes) -> dict[str, ArxivMeta]:
    """Parse an arXiv Atom API response into ArxivMeta keyed by version-less id."""
    root = etree.fromstring(xml)
    out: dict[str, ArxivMeta] = {}
    for entry in root.findall("atom:entry", _NS):
        entry_url = _text(entry, "atom:id")
        title = _text(entry, "atom:title")
        published = _text(entry, "atom:published")
        if not entry_url or not title or not published:
            # arXiv returns an "Error" entry without these for invalid ids.
            logger.warning("skipping arXiv API entry without id/title/published: %s", entry_url)
            continue
        arxiv_id = strip_version(entry_url.split("/abs/")[-1])
        primary = entry.find("arxiv:primary_category", _NS)
        out[arxiv_id] = ArxivMeta(
            arxiv_id=arxiv_id,
            title=title,
            published=datetime.fromisoformat(published.replace("Z", "+00:00")),
            primary_category=primary.get("term", "") if primary is not None else "",
            categories=[c.get("term", "") for c in entry.findall("atom:category", _NS)],
            comment=_text(entry, "arxiv:comment"),
        )
    return out


class ArxivClient:
    """Sequential client that waits `settings.arxiv_delay_s` between real network requests."""

    def __init__(
        self,
        settings: Settings,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        self._sleep = sleep
        self._clock = clock
        self._last_request: float | None = None
        self._http = httpx.Client(
            transport=transport,
            headers={"User-Agent": settings.user_agent},
            follow_redirects=True,
            timeout=httpx.Timeout(60.0, connect=15.0),
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "ArxivClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _throttle(self) -> None:
        if self._last_request is not None:
            wait = self.settings.arxiv_delay_s - (self._clock() - self._last_request)
            if wait > 0:
                logger.debug("arXiv politeness delay %.1fs", wait)
                self._sleep(wait)
        self._last_request = self._clock()

    def _get(self, url: str, params: dict[str, str] | None = None) -> httpx.Response:
        self._throttle()
        try:
            return self._http.get(url, params=params)
        except httpx.HTTPError as e:
            raise ArxivError(f"request to {url} failed: {e}") from e

    def download_pdf(self, arxiv_id: str, dest: Path) -> Path:
        """Download the PDF for `arxiv_id` to `dest` (atomic). Existing files are not re-fetched."""
        if dest.exists():
            logger.info("skip %s: %s exists", arxiv_id, dest)
            return dest
        url = PDF_URL.format(arxiv_id=arxiv_id)
        resp = self._get(url)
        if resp.status_code == 404:
            raise ArxivError(f"404 Not Found: {url}")
        if resp.status_code != 200:
            raise ArxivError(f"HTTP {resp.status_code} for {url}")
        body = resp.content
        if not body.startswith(PDF_MAGIC):
            ctype = resp.headers.get("content-type", "?")
            raise ArxivError(f"not a PDF (content-type {ctype}, {len(body)} bytes): {url}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(body)
        tmp.replace(dest)
        logger.info("fetched %s -> %s (%d KB)", arxiv_id, dest, len(body) // 1024)
        return dest

    def fetch_metadata(self, ids: Iterable[str]) -> dict[str, ArxivMeta]:
        """Look up metadata for `ids` via the arXiv Atom API, keyed by version-less id."""
        id_list = [strip_version(i) for i in ids]
        out: dict[str, ArxivMeta] = {}
        for start in range(0, len(id_list), METADATA_BATCH):
            batch = id_list[start : start + METADATA_BATCH]
            params = {"id_list": ",".join(batch), "max_results": str(len(batch))}
            resp = self._get(API_URL, params=params)
            if resp.status_code != 200:
                # Observed 2026-09-24: export.arxiv.org answers 406 (empty body, from its CDN) to
                # Python TLS clients on cache misses, while curl succeeds. Surface it clearly.
                raise ArxivError(f"arXiv API HTTP {resp.status_code} for {batch}")
            out.update(parse_atom_feed(resp.content))
        missing = [i for i in id_list if i not in out]
        if missing:
            logger.warning("arXiv API returned no metadata for %s", missing)
        return out
