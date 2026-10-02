"""Thin clients for the resolution sources, in lookup order: reviewer-index, OpenAlex, Crossref.

Each returns `Record`s (one published form of a work plus whatever ids/abstract the source has).
Nothing here decides whether a record matches a reference; that is `resolve.py`'s job.
"""

import logging
import re
import threading
from dataclasses import dataclass
from typing import Any

from reviewer.cache import DiskCache
from reviewer.citations.http import CachedJsonClient, SourceError
from reviewer.citations.raw import arxiv_year
from reviewer.config import Settings
from reviewer.index_gate import IndexGateTransport
from reviewer.models import WorkVersion

logger = logging.getLogger(__name__)

OPENALEX_URL = "https://api.openalex.org"
CROSSREF_URL = "https://api.crossref.org/works"
OPENALEX_SELECT = (
    "id,doi,title,publication_year,publication_date,type,authorships,locations,"
    "primary_location,ids,abstract_inverted_index"
)
CROSSREF_SELECT = "DOI,title,author,issued,container-title,type,abstract"
CROSSREF_VENUE_TYPES = frozenset({"journal-article", "proceedings-article", "book-chapter"})
ARXIV_DOI_PREFIX = "10.48550/arxiv."
ARXIV_ABS_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([^\s?#v]+(?:v\d+)?)", re.IGNORECASE)
JATS_TAG_RE = re.compile(r"<[^>]+>")
PREPRINT_DOI_PREFIXES = (
    "10.48550/",  # arXiv
    "10.5281/",  # Zenodo
    "10.21203/",  # Research Square
    "10.31219/",  # OSF
    "10.36227/",  # TechRxiv
    "10.2139/",  # SSRN
    "10.1101/",  # bioRxiv / medRxiv
    "10.20944/",  # Preprints.org
    "10.13140/",  # ResearchGate
    "10.6084/",  # figshare
)


def is_preprint_doi(doi: str | None) -> bool:
    return bool(doi) and doi.lower().startswith(PREPRINT_DOI_PREFIXES)


@dataclass
class Record:
    version: WorkVersion
    arxiv_id: str | None = None
    doi: str | None = None
    """Non-arXiv DOI."""
    openalex_id: str | None = None
    abstract: str | None = None


def _strip_arxiv_version(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id)


# --------------------------------------------------------------------------- reviewer-index


class IndexClient:
    """reviewer-index query API (contract on reviewer-index main):
    `GET /search?q=<text>&k=<int>` -> `[{key, kind, title, abstract, published_date, year,
    similarity}]` and `GET /health` -> `{index_version}`. Probed once; results are cached per
    index_version, so rows added to the index invalidate them."""

    def __init__(self, settings: Settings, http: CachedJsonClient | None = None) -> None:
        self.settings = settings
        self.base = settings.index_url.rstrip("/")
        self._http = http
        self._available: bool | None = None
        self.index_version: str | None = None

    def _key_header(self) -> dict[str, str]:
        key = self.settings.index_api_key
        return {"X-API-Key": key} if key else {}

    def _client(self, namespace: str) -> CachedJsonClient:
        if self._http is None:
            self._http = CachedJsonClient(
                DiskCache(self.settings.cache_dir, namespace, ".json"),
                user_agent=self.settings.user_agent,
                timeout_s=self.settings.index_timeout_s,
                attempts=2,
                headers=self._key_header(),
                # keyed (public) index: share the per-key concurrency limit (index_gate)
                transport=IndexGateTransport() if self.settings.index_api_key else None,
            )
        return self._http

    @property
    def available(self) -> bool:
        if self._available is None and self.settings.index_replay_version is not None:
            # Replay (as in novelty): serve this index version's cached responses; no probe.
            self.index_version = self.settings.index_replay_version
            self._available = True
            self._client(f"citations_index_{self.index_version}")
        if self._available is None:
            probe = self._http or CachedJsonClient(
                DiskCache(self.settings.cache_dir, "citations_index_health", ".json"),
                user_agent=self.settings.user_agent,
                timeout_s=5.0,
                attempts=1,
                headers=self._key_header(),
            )
            try:
                resp = probe.get(f"{self.base}/health", use_cache=False)
            except SourceError as e:
                logger.info("reviewer-index not available at %s (%s)", self.base, e)
                self._available = False
                return False
            body = resp.body if isinstance(resp.body, dict) else {}
            self.index_version = body.get("index_version")
            self._available = resp.status == 200 and bool(self.index_version)
            if self._available:
                self._client(f"citations_index_{self.index_version}")
        return self._available

    def search(self, q: str, k: int = 5) -> list[dict[str, Any]]:
        if not self.available:
            return []
        resp = self._client("citations_index").get(f"{self.base}/search", {"q": q, "k": str(k)})
        if resp.status != 200:
            return []
        body = resp.body
        if isinstance(body, dict):  # tolerate a wrapped response
            body = body.get("results")
        return [h for h in body or [] if isinstance(h, dict)]


# --------------------------------------------------------------------------- OpenAlex


def openalex_abstract(inverted: dict[str, list[int]] | None) -> str | None:
    if not inverted:
        return None
    positions = sorted((pos, word) for word, poss in inverted.items() for pos in poss)
    text = " ".join(word for _, word in positions)
    return text or None


def openalex_records(work: dict[str, Any]) -> list[Record]:
    """One OpenAlex work -> a venue Record and/or an arXiv preprint Record.

    OpenAlex sometimes merges the preprint and venue paper into one work (then both forms are
    returned) and sometimes keeps them as separate works (each returns its own form)."""
    wid = (work.get("id") or "").rsplit("/", 1)[-1]
    title = work.get("title") or ""
    authors = [
        a["author"]["display_name"]
        for a in work.get("authorships") or []
        if (a.get("author") or {}).get("display_name")
    ]
    doi = (work.get("doi") or "").removeprefix("https://doi.org/").lower() or None
    arxiv_id: str | None = None
    if doi and doi.startswith(ARXIV_DOI_PREFIX):
        arxiv_id = doi.removeprefix(ARXIV_DOI_PREFIX)
        doi = None
    venue_name: str | None = None
    for loc in work.get("locations") or []:
        url = loc.get("landing_page_url") or ""
        m = ARXIV_ABS_RE.search(url)
        if m and not arxiv_id:
            arxiv_id = _strip_arxiv_version(m.group(1))
        if url.lower().startswith("https://doi.org/10.48550/arxiv.") and not arxiv_id:
            arxiv_id = url.lower().rsplit("arxiv.", 1)[1]
        source = loc.get("source") or {}
        if (
            source.get("type") in ("journal", "conference")
            and "arxiv" not in (source.get("display_name") or "").lower()
        ):
            venue_name = venue_name or source.get("display_name")
    if is_preprint_doi(doi):
        doi = None
    has_venue = work.get("type") != "preprint" and (venue_name is not None or doi is not None)
    abstract = openalex_abstract(work.get("abstract_inverted_index"))
    out: list[Record] = []
    if has_venue:
        out.append(
            Record(
                WorkVersion(
                    source="openalex",
                    work_id=wid,
                    title=title,
                    year=work.get("publication_year"),
                    publication_date=work.get("publication_date"),
                    venue=venue_name or (f"doi:{doi}" if doi else None),
                    is_preprint=False,
                    authors=authors,
                ),
                arxiv_id=arxiv_id,
                doi=doi,
                openalex_id=wid,
                abstract=abstract,
            )
        )
    if arxiv_id or not has_venue:
        year = arxiv_year(arxiv_id) if arxiv_id else None
        out.append(
            Record(
                WorkVersion(
                    source="openalex",
                    work_id=wid,
                    title=title,
                    year=year or work.get("publication_year"),
                    publication_date=None if has_venue else work.get("publication_date"),
                    venue="arXiv" if arxiv_id else None,
                    is_preprint=True,
                    authors=authors,
                ),
                arxiv_id=arxiv_id,
                doi=doi,
                openalex_id=wid,
                abstract=abstract,
            )
        )
    return out


class OpenAlexClient:
    """Singleton lookups (free) and title search (billed per call). Search stops once the
    remaining daily budget reported by OpenAlex falls to `openalex_min_remaining_usd`, so the
    reviewer-index daily job sharing this key is not starved."""

    def __init__(self, settings: Settings, http: CachedJsonClient | None = None) -> None:
        self.settings = settings
        self.http = http or CachedJsonClient(
            DiskCache(settings.cache_dir, "citations_openalex", ".json"),
            user_agent=settings.user_agent,
            min_interval_s=0.11,
        )
        self._lock = threading.Lock()
        self.remaining_usd: float | None = None
        self.budget_exhausted = False
        self.n_searches = 0

    def _params(self, **extra: str) -> dict[str, str]:
        params = {"select": OPENALEX_SELECT, **extra}
        if self.settings.openalex_api_key:
            params["api_key"] = self.settings.openalex_api_key
        if self.settings.openalex_email:
            params["mailto"] = self.settings.openalex_email
        return params

    def _note_budget(self, headers: dict[str, str]) -> None:
        value = headers.get("x-ratelimit-remaining-usd")
        if value is None:
            return
        with self._lock:
            try:
                self.remaining_usd = float(value)
            except ValueError:
                return
            if self.remaining_usd <= self.settings.openalex_min_remaining_usd:
                self.budget_exhausted = True

    def probe_budget(self) -> None:
        """Read the remaining budget with a free singleton request, so concurrent workers do not
        start paid searches before the first response reports it."""
        try:
            resp = self.http.get(
                f"{OPENALEX_URL}/works/W2741809807", self._params(), use_cache=False
            )
        except SourceError as e:
            logger.warning("OpenAlex budget probe failed: %s", e)
            return
        self._note_budget(resp.headers)

    @property
    def can_search(self) -> bool:
        return not self.budget_exhausted

    def work(self, key: str) -> list[Record]:
        """`key` is 'W…', 'doi:10.…'. 404 -> []."""
        resp = self.http.get(f"{OPENALEX_URL}/works/{key}", self._params())
        self._note_budget(resp.headers)
        if resp.status != 200 or not isinstance(resp.body, dict):
            return []
        return openalex_records(resp.body)

    def by_arxiv(self, arxiv_id: str) -> list[Record]:
        return self.work(f"doi:{ARXIV_DOI_PREFIX}{arxiv_id.lower()}")

    def by_doi(self, doi: str) -> list[Record]:
        return self.work(f"doi:{doi}")

    def search_title(self, title: str, per_page: int = 10) -> list[Record] | None:
        """Title search; cached results are always served. None when a fresh search is skipped
        because the budget floor was reached (so the caller can say so)."""
        # "Let's" would become "Let s"; drop apostrophe words rather than guess the tokenizer.
        kept = [w for w in title.split() if "'" not in w and "’" not in w]
        words = re.sub(r"[^\w\s-]", " ", " ".join(kept)).split()
        if not words:
            return []
        url = f"{OPENALEX_URL}/works"
        params = self._params(filter=f"title.search:{' '.join(words)}", per_page=str(per_page))
        resp = self.http.cached(url, params)
        if resp is None:
            if not self.can_search:
                return None
            try:
                resp = self.http.get(url, params)
            except SourceError as e:
                if "429" in str(e):
                    self.budget_exhausted = True
                    return None
                raise
            self.n_searches += 1
            self._note_budget(resp.headers)
        if resp.status != 200 or not isinstance(resp.body, dict):
            return []
        return [r for w in resp.body.get("results") or [] for r in openalex_records(w)]


# --------------------------------------------------------------------------- Crossref


def crossref_record(item: dict[str, Any]) -> Record:
    title = " ".join((item.get("title") or [""])[0].split())
    authors = []
    for a in item.get("author") or []:
        name = " ".join(p for p in (a.get("given"), a.get("family")) if p) or a.get("name")
        if name:
            authors.append(name)
    parts = ((item.get("issued") or {}).get("date-parts") or [[None]])[0]
    year = parts[0] if parts and isinstance(parts[0], int) else None
    date = "-".join(f"{p:02d}" if i else str(p) for i, p in enumerate(parts) if p) or None
    venue = " ".join((item.get("container-title") or [""])[0].split()) or None
    doi = (item.get("DOI") or "").lower() or None
    abstract = item.get("abstract")
    if abstract:
        abstract = " ".join(JATS_TAG_RE.sub(" ", abstract).split()) or None
    is_preprint = item.get("type") not in CROSSREF_VENUE_TYPES or is_preprint_doi(doi)
    return Record(
        WorkVersion(
            source="crossref",
            work_id=doi or "",
            title=title,
            year=year,
            publication_date=date,
            venue=venue if not is_preprint else (venue or item.get("type")),
            is_preprint=is_preprint,
            authors=authors,
        ),
        doi=None if is_preprint_doi(doi) else doi,
        abstract=abstract,
    )


class CrossrefClient:
    def __init__(self, settings: Settings, http: CachedJsonClient | None = None) -> None:
        self.settings = settings
        self.http = http or CachedJsonClient(
            DiskCache(settings.cache_dir, "citations_crossref", ".json"),
            user_agent=settings.user_agent,
            min_interval_s=0.2,
        )

    def _params(self, **extra: str) -> dict[str, str]:
        params = {"select": CROSSREF_SELECT, **extra}
        if self.settings.openalex_email:
            params["mailto"] = self.settings.openalex_email
        return params

    def bibliographic(self, text: str, rows: int = 5) -> list[Record]:
        resp = self.http.get(
            CROSSREF_URL, self._params(**{"query.bibliographic": text[:500], "rows": str(rows)})
        )
        if resp.status != 200 or not isinstance(resp.body, dict):
            return []
        items = (resp.body.get("message") or {}).get("items") or []
        return [crossref_record(i) for i in items if i.get("title")]

    def by_doi(self, doi: str) -> list[Record]:
        resp = self.http.get(f"{CROSSREF_URL}/{doi}", {})
        if resp.status != 200 or not isinstance(resp.body, dict):
            return []
        return [crossref_record(resp.body.get("message") or {})]
