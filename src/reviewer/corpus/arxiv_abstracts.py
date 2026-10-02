"""arXiv metadata abstracts (the source of truth for arXiv papers), fetched with `curl`.

export.arxiv.org answers 406 to Python TLS clients on cache misses (see arxiv_client.py), while
curl succeeds, so this client shells out to the `curl` executable. Abstracts are cached per id;
a lookup never raises for partial results: ids that could not be fetched are simply absent.
"""

import logging
import subprocess
import time
from collections.abc import Callable, Sequence
from typing import Any
from urllib.parse import urlencode

from lxml import etree
from pydantic import BaseModel

from reviewer.cache import DiskCache, content_key
from reviewer.config import Settings
from reviewer.corpus.arxiv_client import API_URL
from reviewer.corpus.ids import strip_version

logger = logging.getLogger(__name__)

BATCH_SIZE = 20
CACHE_VERSION = "arxiv-abstract-v1"
CURL_MAX_TIME_S = 60
_HTTP_CODE_MARKER = "\n__HTTP_CODE__:"

_NS = {"atom": "http://www.w3.org/2005/Atom"}

Runner = Callable[..., subprocess.CompletedProcess[Any]]


class ArxivAbstract(BaseModel):
    arxiv_id: str
    """Version-less arXiv id."""
    title: str
    abstract: str
    """arXiv's <summary> with whitespace collapsed; LaTeX left as is."""


RETRY_ATTEMPTS = 4
RETRY_BACKOFF_S = 15.0
RETRYABLE_STATUSES = frozenset({"429", "500", "502", "503", "504"})


def _collapse(text: str | None) -> str:
    return " ".join((text or "").split())


def abstract_cache_key(arxiv_id: str) -> str:
    return content_key(CACHE_VERSION, arxiv_id)


def parse_abstracts(xml: bytes) -> dict[str, ArxivAbstract]:
    """Parse an arXiv Atom API response into ArxivAbstract keyed by version-less id."""
    root = etree.fromstring(xml)
    out: dict[str, ArxivAbstract] = {}
    for entry in root.findall("atom:entry", _NS):
        entry_url = _collapse(entry.findtext("atom:id", namespaces=_NS))
        summary = _collapse(entry.findtext("atom:summary", namespaces=_NS))
        if "/abs/" not in entry_url or not summary:
            # Invalid ids come back as an "Error" entry with no /abs/ url.
            logger.warning("skipping arXiv entry without abstract: %s", entry_url or "?")
            continue
        arxiv_id = strip_version(entry_url.split("/abs/")[-1])
        title = _collapse(entry.findtext("atom:title", namespaces=_NS))
        out[arxiv_id] = ArxivAbstract(arxiv_id=arxiv_id, title=title, abstract=summary)
    return out


def curl_args(url: str, user_agent: str) -> list[str]:
    """curl invocation: silent, custom UA, bounded time, HTTP status appended after the body."""
    return [
        "curl",
        "-s",
        "-A",
        user_agent,
        "--max-time",
        str(CURL_MAX_TIME_S),
        "-w",
        _HTTP_CODE_MARKER + "%{http_code}",
        url,
    ]


def query_url(ids: Sequence[str]) -> str:
    return f"{API_URL}?" + urlencode({"id_list": ",".join(ids), "max_results": str(len(ids))})


class ArxivAbstractClient:
    """Batched, cached arXiv abstract lookups over curl, `arxiv_delay_s` apart."""

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
        self.cache = DiskCache(settings.cache_dir, "arxiv_abstract", ".json")

    def _fetch_batch(self, batch: list[str]) -> dict[str, ArxivAbstract]:
        """One batch, retrying HTTP 429/5xx with backoff (arXiv rate-limits bursts)."""
        for attempt in range(RETRY_ATTEMPTS):
            result, status = self._fetch_batch_once(batch)
            if result is not None:
                return result
            if status not in RETRYABLE_STATUSES or attempt == RETRY_ATTEMPTS - 1:
                break
            delay = RETRY_BACKOFF_S * (2**attempt)
            logger.warning("arXiv API HTTP %s; retrying batch in %.0fs", status, delay)
            self._sleep(delay)
        return {}

    def _fetch_batch_once(self, batch: list[str]) -> tuple[dict[str, ArxivAbstract] | None, str]:
        if self._made_request:
            self._sleep(self.settings.arxiv_delay_s)
        self._made_request = True
        url = query_url(batch)
        try:
            proc = self._run(
                curl_args(url, self.settings.user_agent),
                capture_output=True,
                check=False,
                timeout=CURL_MAX_TIME_S + 15,
            )
        except (OSError, subprocess.SubprocessError) as e:
            logger.warning("curl failed for arXiv batch %s: %s", batch, e)
            return None, "curl-error"
        stdout: bytes = proc.stdout or b""
        body, sep, code = stdout.rpartition(_HTTP_CODE_MARKER.encode())
        status = code.decode(errors="replace").strip() if sep else "?"
        if proc.returncode != 0 or status != "200":
            logger.warning(
                "arXiv API batch %s failed: curl exit %s, HTTP %s", batch, proc.returncode, status
            )
            return None, status
        try:
            return parse_abstracts(body), status
        except etree.XMLSyntaxError as e:
            logger.warning("arXiv API batch %s returned unparseable XML: %s", batch, e)
            return None, "bad-xml"

    def lookup(self, ids: Sequence[str]) -> dict[str, ArxivAbstract]:
        """ArxivAbstract per version-less id; cached ids are not re-queried, failures are absent."""
        wanted = list(dict.fromkeys(strip_version(i) for i in ids))
        out: dict[str, ArxivAbstract] = {}
        todo: list[str] = []
        for arxiv_id in wanted:
            cached = self.cache.get_json(abstract_cache_key(arxiv_id))
            if cached is not None:
                out[arxiv_id] = ArxivAbstract.model_validate(cached)
            else:
                todo.append(arxiv_id)
        for start in range(0, len(todo), BATCH_SIZE):
            batch = todo[start : start + BATCH_SIZE]
            fetched = self._fetch_batch(batch)
            for arxiv_id in batch:
                if arxiv_id in fetched:
                    self.cache.put_json(
                        abstract_cache_key(arxiv_id), fetched[arxiv_id].model_dump()
                    )
                    out[arxiv_id] = fetched[arxiv_id]
        missing = [i for i in wanted if i not in out]
        if missing:
            logger.warning("no arXiv abstract for %d id(s): %s", len(missing), missing)
        return out

    def abstracts(self, ids: Sequence[str]) -> dict[str, str]:
        """Abstract text per version-less id (see `lookup`)."""
        return {k: v.abstract for k, v in self.lookup(ids).items()}
