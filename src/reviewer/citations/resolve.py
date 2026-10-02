"""Resolve references to real papers: reviewer-index, then OpenAlex, then Crossref, then arXiv
(printed arXiv id, then title search) for preprints the others miss or mis-title.

A record is accepted if its title matches the raw reference string (fuzzy >= 0.9); authors and
year are *not* used for that (they are what the checks test), except that very short titles
also need an author or year agreement. As a last resort, a record reached through an identifier
printed in the reference (arXiv id or DOI) is accepted despite a different title when at least
two printed surnames and the year agree (papers renamed between arXiv versions). Once a source
resolves a reference, free lookups from the other sources add more versions and abstracts.
"""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from reviewer.citations import raw as rawlib
from reviewer.citations.arxiv_meta import ArxivMetaClient, ArxivRecord
from reviewer.citations.http import SourceError
from reviewer.citations.sources import CrossrefClient, IndexClient, OpenAlexClient, Record
from reviewer.models import CitationSource, Reference, ResolvedWork, WorkVersion

logger = logging.getLogger(__name__)

SHORT_TITLE_WORDS = 3
Lookup = Callable[..., list[Record] | None]
VENUE_TITLE_MIN = 0.9
HINT_TITLE_MIN = 0.8


@dataclass
class Hints:
    arxiv_ids: list[str]
    dois: list[str]
    printed_ids: set[str] = field(default_factory=set)
    """The arXiv ids and DOIs that appear in the raw string (not only in consolidated fields)."""
    title: str | None = None
    """GROBID's title if it matches the raw string (consolidation can replace it), else a
    title guessed from the raw string."""
    title_validated: bool = False
    """True when `title` is GROBID's and matched the raw string."""
    raw_title: str | None = None
    """Title guessed from the raw string's sentence structure."""


def hints_for(ref: Reference) -> Hints:
    raw = ref.raw_string
    arxiv = rawlib.arxiv_ids(raw) + ([ref.arxiv_id] if ref.arxiv_id else [])
    dois = rawlib.dois(raw) + ([ref.doi.lower()] if ref.doi else [])
    arxiv_from_doi = [d.split("arxiv.", 1)[1] for d in dois if d.startswith("10.48550/arxiv.")]
    grobid_ok = bool(ref.title) and rawlib.title_similarity(raw, ref.title or "") >= 0.9
    printed = {a.lower() for a in rawlib.arxiv_ids(raw)} | set(rawlib.dois(raw))
    printed |= {d.split("arxiv.", 1)[1] for d in printed if d.startswith("10.48550/arxiv.")}
    return Hints(
        arxiv_ids=list(dict.fromkeys(a.lower() for a in arxiv + arxiv_from_doi)),
        dois=list(dict.fromkeys(d for d in dois if not d.startswith("10.48550/"))),
        printed_ids=printed,
        title=rawlib.strip_author_prefix(ref.title or "") if grobid_ok else rawlib.raw_title(raw),
        title_validated=grobid_ok,
        raw_title=rawlib.raw_title(raw),
    )


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, rawlib.fold(a), rawlib.fold(b), autojunk=False).ratio()


def _author_overlap(raw: str, record: Record) -> int:
    """How many of the record's authors have a name token printed in the raw string."""
    raw_tokens = set(rawlib.fold(raw).split())
    return sum(1 for a in record.version.authors if rawlib.name_tokens(a) & raw_tokens)


def _author_ok(raw: str, record: Record) -> bool:
    return _author_overlap(raw, record) > 0


def _year_ok(raw: str, record: Record) -> bool:
    year = record.version.year
    return year is not None and any(abs(y - year) <= 1 for y in rawlib.cited_years(raw))


def accept(ref: Reference, hints: Hints, record: Record) -> float | None:
    """Title similarity if `record` is the work `ref` cites, else None."""
    title = record.version.title
    if not title:
        return None
    raw = ref.raw_string
    sim = rawlib.title_similarity(raw, title)
    if sim < rawlib.TITLE_MATCH_MIN:
        # A title printed truncated ("A decomposable attention model" for "... for Natural
        # Language Inference"): accept only with author and year agreement as well.
        printed = rawlib.fold(hints.title or "")
        if (
            len(printed.split()) >= SHORT_TITLE_WORDS
            and rawlib.fold(title).startswith(printed)
            and _author_ok(raw, record)
            and _year_ok(raw, record)
        ):
            return sim
        return None
    # A record whose title is only a fragment of the cited title ("Attention is all you need"
    # for "Attention is all you need in speech separation") matches a window too. GROBID's
    # title can carry author names, so the better of the two title guesses decides.
    guesses = [t for t in (hints.title if hints.title_validated else None, hints.raw_title) if t]
    if guesses and max(_ratio(g, title) for g in guesses) < HINT_TITLE_MIN:
        return None
    short = len(rawlib.fold(title).split()) <= SHORT_TITLE_WORDS
    if short and not (_author_ok(raw, record) or _year_ok(raw, record)):
        return None
    # "The fifth PASCAL RTE challenge" vs "the third ...", "CoNLL-2003" vs "CoNLL-2002".
    if not rawlib.numbers_agree(title, raw):
        return None
    # "Qwen2 Technical Report" vs "Qwen2.5 technical report".
    if not rawlib.versions_agree(title, hints.raw_title or hints.title, raw):
        return None
    # Same-titled re-uploads and proceedings volumes credited to their editors: when both
    # sides name people, at least one must be shared.
    if record.version.authors and rawlib.has_person_names(raw) and not _author_ok(raw, record):
        return None
    return sim


@dataclass
class Resolution:
    ref: Reference
    hints: Hints
    source: CitationSource | None = None
    records: list[tuple[Record, float]] = field(default_factory=list)
    """Accepted records with their title similarity."""
    attempts: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    by_printed_id: list[Record] = field(default_factory=list)
    """Records reached through a printed identifier that failed the title match."""

    @property
    def arxiv_ids(self) -> list[str]:
        ids = [r.arxiv_id for r, _ in self.records if r.arxiv_id] + self.hints.arxiv_ids
        return list(dict.fromkeys(ids)) if self.records else []

    def add(self, records: list[Record], printed_id: bool = False) -> int:
        seen = {
            (r.version.source, r.version.work_id, r.version.is_preprint) for r, _ in self.records
        }
        n = 0
        for rec in records:
            sim = accept(self.ref, self.hints, rec)
            key = (rec.version.source, rec.version.work_id, rec.version.is_preprint)
            if sim is None:
                if printed_id:
                    self.by_printed_id.append(rec)
            elif key not in seen:
                self.records.append((rec, sim))
                seen.add(key)
                n += 1
        return n

    def accept_by_printed_id(self) -> None:
        """Last resort: identifier printed in the reference + >= 2 printed surnames (or the
        only author) + year within 1."""
        raw = self.ref.raw_string
        for rec in self.by_printed_id:
            authors = rec.version.authors
            overlap = _author_overlap(raw, rec)
            if (overlap >= 2 or (overlap == 1 and len(authors) == 1)) and _year_ok(raw, rec):
                sim = rawlib.title_similarity(raw, rec.version.title)
                self.records.append((rec, sim))
                self.attempts.append(
                    f"accepted {rec.version.source} {rec.version.work_id} by printed identifier "
                    f"and {overlap} matching author(s); title differs "
                    f'("{rec.version.title}", match {sim:.2f})'
                )
                if self.source is None:
                    self.source = rec.version.source  # type: ignore[assignment]


def arxiv_record(rec: ArxivRecord) -> Record:
    return Record(
        WorkVersion(
            source="arxiv",
            work_id=rec.arxiv_id,
            title=rec.title,
            year=rec.published.year,
            publication_date=rec.published.date().isoformat(),
            venue="arXiv",
            is_preprint=True,
            authors=rec.authors,
        ),
        arxiv_id=rec.arxiv_id,
        abstract=rec.abstract or None,
    )


class Resolver:
    def __init__(
        self,
        index: IndexClient,
        openalex: OpenAlexClient,
        crossref: CrossrefClient,
        arxiv: ArxivMetaClient,
    ) -> None:
        self.index = index
        self.openalex = openalex
        self.crossref = crossref
        self.arxiv = arxiv

    def _try(
        self, res: Resolution, label: str, fn: Lookup, *args: str, printed_id: bool = False
    ) -> list[Record]:
        try:
            records = fn(*args)
        except SourceError as e:
            res.errors.append(f"{label}: {e}")
            res.attempts.append(f"{label} -> error")
            return []
        if records is None:
            res.attempts.append(f"{label} -> skipped (OpenAlex budget floor reached)")
            return []
        n = res.add(records, printed_id=printed_id)
        res.attempts.append(f"{label} -> {len(records)} record(s), {n} accepted")
        return records

    # ----------------------------------------------------------------------- sources

    def _index(self, res: Resolution) -> None:
        """reviewer-index search by title. A matching hit is a version in its own right (it
        carries the abstract); the free OpenAlex record behind it adds authors and venue."""
        if not self.index.available or not rawlib.looks_like_reference(res.ref.raw_string):
            return
        # Title first. Embedding search tolerates author names, so the raw string (minus
        # identifiers) is the fallback query when there is no title or the title found nothing;
        # hits are validated against the raw string either way.
        raw_query = " ".join(rawlib.strip_identifiers(res.ref.raw_string).split())
        queries = [q for q in (res.hints.title, raw_query) if q]
        for query in dict.fromkeys(queries):
            if self._index_query(res, query) is None or res.records:
                return

    def _index_query(self, res: Resolution, query: str) -> int | None:
        """One index search; the number of records accepted, or None on error."""
        try:
            hits = self.index.search(query, k=5)
        except SourceError as e:
            res.errors.append(f"index search: {e}")
            res.attempts.append("index search -> error")
            return None
        n_before = len(res.records)
        for hit in hits:
            title = hit.get("title") or ""
            if rawlib.title_similarity(res.ref.raw_string, title) < rawlib.TITLE_MATCH_MIN:
                continue
            key = str(hit.get("key") or "")
            kind = hit.get("kind")
            year = hit.get("year")
            arxiv_id = key.removeprefix("arxiv:") if kind == "arxiv" else None
            res.add(
                [
                    Record(
                        WorkVersion(
                            source="index",
                            work_id=key,
                            title=title,
                            year=year if isinstance(year, int) else None,
                            publication_date=hit.get("published_date"),
                            venue="arXiv" if kind == "arxiv" else f"reviewer-index {kind}",
                            # Only the index's `venue` kind is a venue publication; `external`
                            # works are often the arXiv preprint itself. Venue evidence comes
                            # from the OpenAlex record fetched below.
                            is_preprint=kind != "venue",
                        ),
                        arxiv_id=arxiv_id,
                        openalex_id=key.split(":", 1)[1] if key.startswith("openalex:") else None,
                        abstract=hit.get("abstract") or None,
                    )
                ]
            )
            if key.startswith("openalex:"):
                self._try(res, f"index {key} -> openalex", self.openalex.work, key.split(":", 1)[1])
            elif arxiv_id:
                self._try(res, f"index {key} -> openalex", self.openalex.by_arxiv, arxiv_id)
        n = len(res.records) - n_before
        res.attempts.append(
            f"index search '{query[:60]}' -> {len(hits)} hit(s), {n} record(s) accepted"
        )
        return n

    def _openalex_free(self, res: Resolution) -> None:
        printed = res.hints.printed_ids
        for a in res.hints.arxiv_ids:
            self._try(
                res, f"openalex arxiv:{a}", self.openalex.by_arxiv, a, printed_id=a in printed
            )
        for d in res.hints.dois:
            self._try(res, f"openalex doi:{d}", self.openalex.by_doi, d, printed_id=d in printed)

    def _openalex_search(self, res: Resolution) -> None:
        if res.hints.title:
            self._try(res, "openalex title.search", self.openalex.search_title, res.hints.title)

    def _arxiv_ids(self, ids: list[str]) -> list[Record]:
        return [arxiv_record(r) for r in self.arxiv.lookup(ids).values()]

    def _arxiv_title(self, title: str) -> list[Record] | None:
        found = self.arxiv.search_title(title)
        if found is None:
            raise SourceError("arXiv title search failed")
        return [arxiv_record(r) for r in found]

    def _arxiv_supplement(self, res: Resolution) -> None:
        """Add the arXiv original to a work resolved only to non-arXiv preprint copies."""
        if res.hints.arxiv_ids:
            self._try(res, f"arxiv id {res.hints.arxiv_ids}", self._arxiv_ids, res.hints.arxiv_ids)
        elif res.hints.title:
            self._try(res, "arxiv title search (supplement)", self._arxiv_title, res.hints.title)

    def _arxiv(self, res: Resolution) -> None:
        printed = [a for a in res.hints.arxiv_ids if a in res.hints.printed_ids]
        others = [a for a in res.hints.arxiv_ids if a not in res.hints.printed_ids]
        if printed:
            self._try(res, f"arxiv id {printed}", self._arxiv_ids, printed, printed_id=True)
        if others and not res.records:
            self._try(res, f"arxiv id {others}", self._arxiv_ids, others)
        if not res.records and res.hints.title:
            self._try(res, "arxiv title search", self._arxiv_title, res.hints.title)

    def _crossref(self, res: Resolution) -> None:
        self._try(
            res, "crossref bibliographic(raw)", self.crossref.bibliographic, res.ref.raw_string
        )
        if not res.records:
            for d in res.hints.dois:
                printed = d in res.hints.printed_ids
                self._try(res, f"crossref doi:{d}", self.crossref.by_doi, d, printed_id=printed)

    def resolve(self, ref: Reference) -> Resolution:
        res = Resolution(ref=ref, hints=hints_for(ref))
        self._index(res)
        if res.records:
            res.source = "index"
            self._openalex_free(res)  # free: more versions, abstracts, authors
            return res
        self._openalex_free(res)
        if not res.records:
            self._openalex_search(res)
        if res.records:
            res.source = "openalex"
            return res
        self._crossref(res)
        if res.records:
            res.source = "crossref"
            # Crossref also holds late repository copies of famous preprints (e.g. 2025
            # "posted-content" copies of "Attention Is All You Need"); if that is all it found,
            # look for the arXiv original too.
            if not any(r.arxiv_id or not r.version.is_preprint for r, _ in res.records):
                self._arxiv_supplement(res)
            return res
        self._arxiv(res)
        if res.records:
            res.source = "arxiv"
            return res
        res.accept_by_printed_id()
        return res

    # ----------------------------------------------------------------------- more versions

    def find_versions(
        self, res: Resolution, arxiv: ArxivRecord | None, *, year_mismatch: bool
    ) -> None:
        """Look for further versions of a resolved work: arXiv's journal_ref / DOI (free), then
        Crossref (free), then OpenAlex title search (budgeted).

        Without `year_mismatch` this only runs until a venue version is found (for the
        arXiv-now-published check). With it, every source is asked, and preprint versions are
        kept too, so a year is flagged only if no known version has it."""
        if not res.records:
            return

        def has_venue() -> bool:
            return any(not r.version.is_preprint for r, _ in res.records)

        def done() -> bool:
            return not year_mismatch and has_venue()

        if done():
            return
        title, first = self._title_and_first_author(res, arxiv)
        if arxiv is not None and arxiv.doi:
            label = f"crossref doi:{arxiv.doi} (arXiv metadata)"
            for rec in self._try_more(res, label, self.crossref.by_doi, arxiv.doi):
                self._add_version(res, rec, title, first, year_mismatch)
        if arxiv is not None and arxiv.journal_ref and not has_venue():
            years = rawlib.cited_years(arxiv.journal_ref)
            version = WorkVersion(
                source="arxiv",
                work_id=arxiv.arxiv_id,
                title=arxiv.title,
                year=years[-1] if years else None,
                venue=f"journal_ref: {arxiv.journal_ref}",
                is_preprint=False,
                authors=arxiv.authors,
            )
            res.records.append((Record(version, arxiv_id=arxiv.arxiv_id), 1.0))
            res.attempts.append(f"arXiv journal_ref '{arxiv.journal_ref}' -> venue version")
        if done():
            return
        query = f"{title} {first}".strip()
        for rec in self._try_more(
            res, "crossref version search", self.crossref.bibliographic, query
        ):
            self._add_version(res, rec, title, first, year_mismatch)
        if done():
            return
        found = self._try_more(
            res, "openalex version title.search", self.openalex.search_title, title
        )
        for rec in found:
            self._add_version(res, rec, title, first, year_mismatch)

    def _try_more(self, res: Resolution, label: str, fn: Lookup, *args: str) -> list[Record]:
        try:
            records = fn(*args)
        except SourceError as e:
            res.errors.append(f"{label}: {e}")
            res.attempts.append(f"{label} -> error")
            return []
        if records is None:
            res.attempts.append(f"{label} -> skipped (OpenAlex budget floor reached)")
            return []
        res.attempts.append(f"{label} -> {len(records)} record(s)")
        return records

    def find_arxiv_version(self, res: Resolution) -> None:
        """For a resolved work with no arXiv id (e.g. a NeurIPS record without an abstract),
        look for its arXiv preprint by title (free): it brings the abstract and preprint year.
        Accepted only with a title match and the same first author."""
        if not res.records or res.arxiv_ids:
            return
        title, first = self._title_and_first_author(res, None)
        found = self._try_more(res, "arxiv version title search", self._arxiv_title, title)
        for rec in found:
            self._add_version(res, rec, title, first, keep_preprints=True)

    @staticmethod
    def _title_and_first_author(res: Resolution, arxiv: ArxivRecord | None) -> tuple[str, str]:
        best = anchor(res)
        title = arxiv.title if arxiv else best.version.title
        authors = arxiv.authors if arxiv and arxiv.authors else best.version.authors
        return title, rawlib.surname(authors[0]) if authors else ""

    @staticmethod
    def _add_version(
        res: Resolution, rec: Record, title: str, first: str, keep_preprints: bool
    ) -> None:
        v = rec.version
        if (v.is_preprint and not keep_preprints) or _ratio(title, v.title) < VENUE_TITLE_MIN:
            return
        if first and v.authors and first not in {rawlib.surname(a) for a in v.authors}:
            return
        key = (v.source, v.work_id, v.is_preprint)
        if key in {
            (r.version.source, r.version.work_id, r.version.is_preprint) for r, _ in res.records
        }:
            return
        res.records.append((rec, _ratio(title, v.title)))
        kind = "preprint" if v.is_preprint else "venue"
        res.attempts.append(
            f"{kind} version accepted: {v.source} {v.work_id} ({v.venue}, {v.year})"
        )


def _raw_author_overlap(raw: str, record: Record) -> int:
    tokens = set(rawlib.fold(raw).split())
    return sum(1 for a in record.version.authors if rawlib.surname(a) in tokens)


def anchor(res: Resolution) -> Record:
    """The accepted record most likely to be the cited work: most printed surnames in common
    with the raw string, then best title match, then venue over preprint."""
    return max(
        res.records,
        key=lambda rs: (
            _raw_author_overlap(res.ref.raw_string, rs[0]),
            rs[1],
            not rs[0].version.is_preprint,
        ),
    )[0]


def prune_to_anchor(res: Resolution) -> list[str]:
    """Drop accepted records whose authors share nothing with the anchor's (title-only matches
    to a different work, e.g. a mis-attributed OpenAlex record). Returns what was dropped."""
    if len(res.records) < 2:
        return []
    a = anchor(res)
    a_tokens: set[str] = set()
    for name in a.version.authors:
        a_tokens |= rawlib.name_tokens(name)
    if not a_tokens:
        return []
    kept, dropped = [], []
    for rec, sim in res.records:
        names = rec.version.authors
        if rec is a or not names or any(rawlib.surname(n) in a_tokens for n in names):
            kept.append((rec, sim))
        else:
            v = rec.version
            dropped.append(
                f"dropped {v.source} {v.work_id} ({v.venue}): no author shared with the anchor"
            )
    res.records = kept
    res.attempts += dropped
    return dropped


# --------------------------------------------------------------------------- assembly

WORKSHOP_RE = re.compile(r"\bworkshops?\b", re.IGNORECASE)


def build_resolved(
    res: Resolution, arxiv_records: dict[str, ArxivRecord], abstract_chars: int
) -> ResolvedWork | None:
    if not res.records or res.source is None:
        return None
    prune_to_anchor(res)
    versions: list[WorkVersion] = []
    seen: set[tuple[str, str, bool]] = set()
    for rec, _ in res.records:
        key = (rec.version.source, rec.version.work_id, rec.version.is_preprint)
        if key not in seen:
            seen.add(key)
            versions.append(rec.version)
    arxiv = matching_arxiv(res, arxiv_records)
    arxiv_id = arxiv.arxiv_id if arxiv else next(iter(res.arxiv_ids), None)
    if arxiv is not None:
        versions.append(
            WorkVersion(
                source="arxiv",
                work_id=arxiv.arxiv_id,
                title=arxiv.title,
                year=arxiv.published.year,
                publication_date=arxiv.published.date().isoformat(),
                venue="arXiv",
                is_preprint=True,
                authors=arxiv.authors,
            )
        )
        if arxiv.updated and arxiv.updated.year != arxiv.published.year:
            versions.append(
                WorkVersion(
                    source="arxiv",
                    work_id=arxiv.arxiv_id,
                    title=arxiv.title,
                    year=arxiv.updated.year,
                    publication_date=arxiv.updated.date().isoformat(),
                    venue="arXiv (latest version)",
                    is_preprint=True,
                    authors=arxiv.authors,
                )
            )
    versions = list({(v.source, v.work_id, v.is_preprint, v.year): v for v in versions}.values())
    best_rec = anchor(res)
    best = next(rs for rs in res.records if rs[0] is best_rec)
    authors = (arxiv.authors if arxiv else []) or max(
        (r.version.authors for r, _ in res.records), key=len
    )
    abstract, abstract_source = _pick_abstract(res, arxiv)
    if abstract and len(abstract) > abstract_chars:
        abstract = abstract[:abstract_chars].rsplit(" ", 1)[0] + " …"
    return ResolvedWork(
        source=res.source,
        title=best[0].version.title,
        authors=authors,
        arxiv_id=arxiv_id,
        doi=next((r.doi for r, _ in res.records if r.doi), None),
        openalex_ids=list(dict.fromkeys(r.openalex_id for r, _ in res.records if r.openalex_id)),
        versions=versions,
        title_similarity=round(best[1], 3),
        abstract=abstract,
        abstract_source=abstract_source,
    )


def matching_arxiv(res: Resolution, arxiv_records: dict[str, ArxivRecord]) -> ArxivRecord | None:
    """The first arXiv record for this reference whose title matches the raw string (an arXiv
    id from a consolidated field can point at a different paper)."""
    for arxiv_id in res.arxiv_ids:
        rec = arxiv_records.get(arxiv_id)
        if rec and rawlib.title_similarity(res.ref.raw_string, rec.title) >= rawlib.TITLE_MATCH_MIN:
            return rec
    return None


def _pick_abstract(res: Resolution, arxiv: ArxivRecord | None) -> tuple[str | None, str | None]:
    """arXiv's abstract is authoritative; OpenAlex's is reconstructed; Crossref's is rare."""
    if arxiv is not None and arxiv.abstract.strip():
        return arxiv.abstract, "arxiv"
    for rec, _ in res.records:
        if rec.abstract and rec.version.source == "arxiv":
            return rec.abstract, "arxiv"
    for rec, _ in res.records:
        if rec.abstract and rec.version.source == "index":
            return rec.abstract, "index"
    for rec, _ in res.records:
        if rec.abstract and rec.version.source == "openalex":
            return rec.abstract, "openalex"
    for rec, _ in res.records:
        if rec.abstract and rec.version.source == "crossref":
            return rec.abstract, "crossref"
    return None, None


def is_workshop(version: WorkVersion) -> bool:
    return bool(version.venue and WORKSHOP_RE.search(version.venue))
