"""Reference flags. Pure functions: they compare the raw string (what the paper prints) with the
resolved record (what the sources say). No network."""

import re
from datetime import date

from reviewer.citations import raw as rawlib
from reviewer.citations.resolve import is_workshop
from reviewer.models import CitationFlag, FlagType, ReferenceCheck, ResolvedWork, WorkVersion

AUTHOR_CHECK_TITLE_MIN = 0.97
MAX_NAME_TOKENS = 4
ORG_WORDS = frozenset(
    {
        "team",
        "group",
        "consortium",
        "openai",
        "google",
        "meta",
        "deepmind",
        "anthropic",
        "collaboration",
        "contributors",
        "inc",
        "ltd",
    }
)


def _version_line(v: WorkVersion) -> str:
    when = v.publication_date or (str(v.year) if v.year else "year ?")
    kind = "preprint" if v.is_preprint else "venue"
    return f"{v.source} {v.work_id} — {kind}: {v.venue or '?'} ({when})"


def _record_lines(resolved: ResolvedWork) -> list[str]:
    match = f"title match {resolved.title_similarity:.2f}"
    lines = [f'resolved via {resolved.source}: "{resolved.title}" ({match})']
    lines += [f"version: {_version_line(v)}" for v in resolved.versions]
    return lines


BUDGET_SKIP = "skipped (OpenAlex budget"
SEARCHED = ("index search", "openalex", "arxiv")
"""Attempt prefixes that mean a title or identifier search ran (Crossref's free-text
bibliographic query alone is too weak for ML papers to call a reference unresolvable)."""


def not_checked_reason(check: ReferenceCheck) -> str | None:
    """Why an unresolved reference is not flagged: a web page rather than a paper, or a lookup
    that was skipped (budget) or failed, so absence of a match is not evidence."""
    if check.resolved is not None:
        return None
    if not rawlib.looks_like_reference(check.raw_string):
        return "not a reference (parse artifact: no year, identifier or venue)"
    if rawlib.is_web_resource(check.raw_string):
        return "web resource (URL, no paper identifiers)"
    if not any(a.startswith(SEARCHED) for a in check.attempts):
        return "no title or identifier could be parsed to search with"
    if any(BUDGET_SKIP in a for a in check.attempts):
        return "OpenAlex title search skipped (daily budget floor)"
    if any(a.endswith("-> error") for a in check.attempts):
        return "a lookup failed"
    return None


def unresolved_flag(check: ReferenceCheck) -> CitationFlag | None:
    if check.resolved is not None or not_checked_reason(check):
        return None
    return CitationFlag(
        type=FlagType.unresolved,
        ref_id=check.ref_id,
        detail="no source returned a record whose title matches the reference",
        evidence=[f"raw: {check.raw_string}", *(f"lookup: {a}" for a in check.attempts)],
    )


def year_flag(check: ReferenceCheck) -> CitationFlag | None:
    """Flag a printed year that the version the reference points to does not have.

    - arXiv-only citation: compare with the arXiv span (v1 .. latest version).
    - Venue citation whose venue we hold (matched by name/alias): compare with that venue
      version's year, also accepting the arXiv span (citing the preprint year is fine).
    - Venue citation whose venue we do not hold: not flagged. ML venues are thinly covered and
      we rarely hold the earliest version of older works, so no held version can decide it.
    Other held versions (reprints, book chapters, journal extensions) are shown as evidence but
    never decide the flag: a reference may point to a version we lack."""
    r = check.resolved
    if r is None or not check.cited_years:
        return None
    cited = set(check.cited_years)
    arxiv_years = {v.year for v in r.versions if v.source == "arxiv" and v.year}
    if r.arxiv_id and (y := rawlib.arxiv_year(r.arxiv_id)):
        arxiv_years.add(y)
    arxiv_span = set(range(min(arxiv_years), max(arxiv_years) + 1)) if arxiv_years else set()
    if cited & arxiv_span:
        return None
    if check.cited_as_arxiv_only:
        if not arxiv_span:
            return None
        known, reason = arxiv_span, "outside the arXiv versions"
    else:
        printed = rawlib.venue_keys(check.raw_string)
        held = [
            v for v in r.venue_versions if v.year and printed & rawlib.venue_keys(v.venue or "")
        ]
        if held:
            known = {v.year for v in held if v.year} | arxiv_span
            if cited & known:
                return None
            reason = f"not the year of the {held[0].venue} version"
        else:
            return None
    return CitationFlag(
        type=FlagType.wrong_year,
        ref_id=check.ref_id,
        detail=(
            f"cites {', '.join(map(str, sorted(cited)))}, {reason} "
            f"({', '.join(map(str, sorted(known)))})"
        ),
        evidence=[f"raw: {check.raw_string}", *_record_lines(r)],
    )


def _is_org(name: str) -> bool:
    return any(t in ORG_WORDS for t in name.split())


def author_flag(check: ReferenceCheck) -> CitationFlag | None:
    """Flag when the first printed author, or more than half of the printed authors, match no
    author of any known version. Name order and given/family order are ignored."""
    r = check.resolved
    if r is None:
        return None
    if r.title_similarity < AUTHOR_CHECK_TITLE_MIN:
        return None
    names, et_al = rawlib.author_chunks(check.raw_string, r.title)
    # A year inside the author segment (beyond an author-year style trailing year) means
    # GROBID merged two references: the names may belong to the previous one.
    segment = " ".join(names)
    if len(rawlib.YEAR_RE.findall(segment)) > 0:
        return None
    names = [n for n in names if not _is_org(n) and len(n.split()) <= MAX_NAME_TOKENS]
    record_names = list(dict.fromkeys([*r.authors, *(a for v in r.versions for a in v.authors)]))
    if not names or not record_names:
        return None
    tokens: set[str] = set()
    for a in record_names:
        tokens |= rawlib.name_tokens(a)
    unmatched = [n for n in names if not (rawlib.name_tokens(n) & tokens)]
    if not unmatched:
        return None
    first_bad = unmatched[0] == names[0]
    # A record listing fewer authors than printed is usually incomplete metadata, so the
    # fraction rule needs a record at least as long as the printed list.
    many_bad = (
        len(names) >= 2 and len(record_names) >= len(names) and len(unmatched) / len(names) > 0.5
    )
    if not (first_bad or many_bad):
        return None
    shown = ", ".join(r.authors[:8]) + (" …" if len(r.authors) > 8 else "")
    reason = "first author" if first_bad else f"{len(unmatched)}/{len(names)} printed authors"
    return CitationFlag(
        type=FlagType.wrong_authors,
        ref_id=check.ref_id,
        detail=f"{reason} not found among the record's authors: {', '.join(unmatched)}",
        evidence=[
            f"raw: {check.raw_string}",
            f"printed authors{' (et al.)' if et_al else ''}: {', '.join(names)}",
            f"record authors: {shown}",
            *_record_lines(r),
        ],
    )


def citing_date(paper_id: str) -> date | None:
    """First day of the citing paper's arXiv submission month (None for non-arXiv ids)."""
    m = re.fullmatch(r"(\d{2})(\d{2})\.\d{4,5}", paper_id)
    if not m or not 1 <= int(m.group(2)) <= 12:
        return None
    return date(2000 + int(m.group(1)), int(m.group(2)), 1)


def _month(v: WorkVersion) -> tuple[int, int] | None:
    """(year, month) when the source gives a real month. OpenAlex writes year-only dates as
    YYYY-01-01, so a January 1st date is treated as year-only."""
    d = v.publication_date or ""
    if len(d) < 7 or d.endswith("-01-01") or (d.endswith("-01") and len(d) == 7):
        return None
    return int(d[:4]), int(d[5:7])


def _timing(v: WorkVersion, cited_on: date | None) -> str:
    if cited_on is None or v.year is None:
        return "timing unknown"
    month = _month(v)
    if month is not None:
        before = month < (cited_on.year, cited_on.month)
        return f"venue version {'predates' if before else 'appeared after'} the citing paper"
    if v.year < cited_on.year:
        return "venue version predates the citing paper"
    if v.year > cited_on.year:
        return "venue version appeared after the citing paper"
    return "venue version from the same year as the citing paper"


def arxiv_published_flag(check: ReferenceCheck, cited_on: date | None) -> CitationFlag | None:
    r = check.resolved
    if r is None or not check.cited_as_arxiv_only:
        return None
    venues = [v for v in r.venue_versions if not is_workshop(v)]
    if not venues:
        return None
    v = min(venues, key=lambda x: x.year or 9999)
    return CitationFlag(
        type=FlagType.arxiv_now_published,
        ref_id=check.ref_id,
        detail=f"cited as arXiv only; published at {v.venue} ({v.year}); {_timing(v, cited_on)}",
        evidence=[f"raw: {check.raw_string}", *_record_lines(r)],
    )


def _identity_keys(check: ReferenceCheck) -> set[str]:
    r = check.resolved
    if r is None:
        return set()
    keys = {f"oa:{w}" for w in r.openalex_ids}
    if r.arxiv_id:
        keys.add(f"arxiv:{r.arxiv_id}")
    if r.doi:
        keys.add(f"doi:{r.doi}")
    keys.add(f"title:{rawlib.fold(r.title)}")
    return keys


def duplicate_flags(checks: list[ReferenceCheck]) -> list[CitationFlag]:
    """Flag every later reference that shares an identity (arXiv id, DOI, OpenAlex id, or
    normalized resolved title) with an earlier one."""
    owner: dict[str, str] = {}
    by_id = {c.ref_id: c for c in checks}
    dup_of: dict[str, tuple[str, str]] = {}
    for c in checks:
        for key in sorted(_identity_keys(c)):
            if key in owner and owner[key] != c.ref_id and c.ref_id not in dup_of:
                dup_of[c.ref_id] = (owner[key], key)
            owner.setdefault(key, c.ref_id)
    flags: list[CitationFlag] = []
    for ref_id, (first, key) in dup_of.items():
        flags.append(
            CitationFlag(
                type=FlagType.duplicate,
                ref_id=ref_id,
                detail=f"same work as {first} (shared {key.split(':', 1)[0]})",
                evidence=[
                    f"{ref_id} raw: {by_id[ref_id].raw_string}",
                    f"{first} raw: {by_id[first].raw_string}",
                    f"shared identity: {key}",
                ],
            )
        )
    return flags


def run_checks(checks: list[ReferenceCheck], paper_id: str) -> None:
    """Attach flags to each check in place."""
    cited_on = citing_date(paper_id)
    for c in checks:
        c.flags = [
            f
            for f in (
                unresolved_flag(c),
                year_flag(c),
                author_flag(c),
                arxiv_published_flag(c, cited_on),
            )
            if f is not None
        ]
    by_id = {c.ref_id: c for c in checks}
    for f in duplicate_flags(checks):
        by_id[f.ref_id].flags.append(f)
