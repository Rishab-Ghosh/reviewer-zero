"""Thin, cached client for the GROBID fulltext endpoint."""

import logging
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import httpx

from reviewer.cache import DiskCache, content_key, sha256_file
from reviewer.config import Settings

logger = logging.getLogger(__name__)

FULLTEXT_PATH = "/api/processFulltextDocument"
ISALIVE_PATH = "/api/isalive"

# TEI elements GROBID should annotate with `coords="page,x,y,w,h;..."`, so parsed text can be
# attributed to PDF pages (see tei.page_texts). Sent as a repeated `teiCoordinates` form field.
TEI_COORDINATES: tuple[str, ...] = (
    "s",
    "head",
    "p",
    "formula",
    "figure",
    "ref",
    "biblStruct",
    "note",
    "persName",
    "title",
    "affiliation",
)

type GrobidParams = Mapping[str, str | list[str]]

# Request parameters; part of the cache key, so changing them invalidates cached TEI.
# A list value is sent as a repeated form field.
GROBID_PARAMS: dict[str, str | list[str]] = {
    "consolidateHeader": "1",
    "consolidateCitations": "1",
    "includeRawCitations": "1",
    "segmentSentences": "1",
    "teiCoordinates": list(TEI_COORDINATES),
}
MAX_BUSY_RETRIES = 3
BUSY_BACKOFF_S = 2.0
# A dropped connection usually means the GROBID container restarted (e.g. memory pressure on a
# large PDF). Wait for it to come back and retry that paper once, so one crash does not cascade
# into failures for every remaining paper.
MAX_CONNECTION_RETRIES = 1
RESTART_WAIT_S = 180.0
RESTART_POLL_S = 5.0


class GrobidError(RuntimeError):
    """GROBID returned an error or could not be reached."""


def canonical_params(params: GrobidParams) -> str:
    """Order-independent string form of the request parameters, used in the cache key.

    Repeated (list) values are sorted, so their order does not matter either.
    """
    parts: list[str] = []
    for key in sorted(params):
        value = params[key]
        values = sorted(value) if isinstance(value, list) else [value]
        parts.extend(f"{key}={v}" for v in values)
    return "&".join(parts)


def grobid_cache_key(pdf_sha256: str, params: GrobidParams = GROBID_PARAMS) -> str:
    return content_key(pdf_sha256, canonical_params(params))


class GrobidClient:
    """Posts PDFs to GROBID and caches the TEI by PDF content hash + request parameters."""

    def __init__(
        self,
        settings: Settings,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base_url = settings.grobid_url.rstrip("/")
        self.cache = DiskCache(settings.paper_cache, "grobid", ".tei.xml")
        self._client = httpx.Client(
            base_url=self.base_url, timeout=settings.grobid_timeout_s, transport=transport
        )
        self._sleep = sleep

    def close(self) -> None:
        self._client.close()

    def is_alive(self) -> bool:
        try:
            resp = self._client.get(ISALIVE_PATH, timeout=5.0)
        except httpx.HTTPError as exc:
            logger.debug("GROBID isalive failed: %s", exc)
            return False
        return resp.status_code == 200 and resp.text.strip().lower() == "true"

    def wait_until_alive(self, max_wait_s: float = RESTART_WAIT_S) -> bool:
        waited = 0.0
        while waited < max_wait_s:
            if self.is_alive():
                return True
            self._sleep(RESTART_POLL_S)
            waited += RESTART_POLL_S
        return self.is_alive()

    def process(self, pdf: Path) -> str:
        """Return TEI XML for `pdf`, from cache when available."""
        key = grobid_cache_key(sha256_file(pdf))
        cached = self.cache.get_text(key)
        if cached is not None:
            return cached
        tei = self._post(pdf)
        self.cache.put_text(key, tei)
        return tei

    def _post(self, pdf: Path) -> str:
        pdf_bytes = pdf.read_bytes()
        connection_retries = 0
        attempt = 0
        while attempt <= MAX_BUSY_RETRIES:
            started = time.monotonic()
            try:
                resp = self._client.post(
                    FULLTEXT_PATH,
                    files={"input": (pdf.name, pdf_bytes, "application/pdf")},
                    data=GROBID_PARAMS,
                )
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError) as exc:
                if connection_retries < MAX_CONNECTION_RETRIES:
                    connection_retries += 1
                    logger.warning(
                        "GROBID dropped the connection on %s (%r); waiting for it to come back",
                        pdf.name,
                        exc,
                    )
                    if self.wait_until_alive():
                        continue
                raise GrobidError(f"GROBID request failed for {pdf.name}: {exc!r}") from exc
            except httpx.HTTPError as exc:
                raise GrobidError(f"GROBID request failed for {pdf.name}: {exc!r}") from exc
            if resp.status_code == 200:
                logger.info("GROBID parsed %s in %.1fs", pdf.name, time.monotonic() - started)
                return resp.text
            if resp.status_code == 503 and attempt < MAX_BUSY_RETRIES:
                delay = BUSY_BACKOFF_S * (attempt + 1)
                logger.warning("GROBID busy (503) on %s; retrying in %.0fs", pdf.name, delay)
                self._sleep(delay)
                attempt += 1
                continue
            raise GrobidError(
                f"GROBID returned HTTP {resp.status_code} for {pdf.name}: {resp.text[:200]!r}"
            )
        raise GrobidError(f"GROBID busy after {MAX_BUSY_RETRIES} retries for {pdf.name}")
