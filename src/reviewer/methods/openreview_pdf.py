"""Download an OpenReview submission's PDF as its reviewers saw it.

OpenReview blocks anonymous API access (403 ChallengeRequiredError), so this logs in with
OPENREVIEW_USERNAME / OPENREVIEW_PASSWORD (token kept in memory only). Revision histories are
cached on disk (keyed by request, never by credentials); the chosen PDF is saved to
data/openreview/pdf/<submission id>.pdf and never downloaded again.

Which revision: the latest PDF revision made before the submission deadline plus
`REVIEW_GRACE` (reviewers see the deadline version; later revisions answer the reviews, and a
checker run on them would miss concerns the authors already fixed). If the history has no such
revision, the earliest PDF revision; if there is no history at all, the current PDF. The choice
and its reason are returned so the eval can report them.

Checked live on 2026-09-26: the legacy API (ICLR 2023) returns the reference history with the
deadline PDF. API v2 (ICLR 2024/25, NeurIPS 2023/24) returns no edits to our account, so there
the current PDF is used, and its own CreationDate (pdfinfo) records whether it is still the
deadline version or a later revision (`provenance`), which the eval reports separately.

API shapes (v2 /notes/edits + /notes/edits/attachment; v1 /references + /references/pdf) are
coded from the OpenReview API docs and tested against recorded-shape fakes only.
"""

import json
import logging
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx

from reviewer.cache import DiskCache, content_key
from reviewer.config import Settings

logger = logging.getLogger(__name__)

API = "https://api2.openreview.net"
API_V1 = "https://api.openreview.net"
REVIEW_GRACE = timedelta(days=7)
V1_LAST_YEAR = {"ICLR.cc": 2023}
"""Venues on the legacy API up to and including this year (ICLR 2023 is v1-only)."""


class OpenReviewError(RuntimeError):
    pass


@dataclass(frozen=True)
class Revision:
    id: str
    """Edit id (API v2) or reference id (API v1)."""
    tcdate_ms: int
    has_pdf: bool


@dataclass(frozen=True)
class PdfChoice:
    revision_id: str | None
    """None: the current PDF (no revision history)."""
    reason: str
    kind: str = "current"
    """'deadline' (latest before deadline + grace), 'earliest', or 'current' (no history)."""


def api_for(venue: str) -> str:
    group, _, rest = venue.partition("/")
    year = int(rest.split("/")[0]) if rest[:4].isdigit() else None
    last = V1_LAST_YEAR.get(group)
    return API_V1 if last is not None and year is not None and year <= last else API


def choose_revision(revisions: list[Revision], deadline: date | None) -> PdfChoice:
    """Pure: the revision reviewers saw (see module docstring)."""
    pdfs = sorted((r for r in revisions if r.has_pdf), key=lambda r: r.tcdate_ms)
    if not pdfs:
        return PdfChoice(None, "no PDF revisions visible in history; current PDF")
    if deadline is None:
        return PdfChoice(pdfs[0].id, "no deadline known; earliest PDF revision", "earliest")
    limit = datetime(deadline.year, deadline.month, deadline.day, tzinfo=UTC) + REVIEW_GRACE
    before = [r for r in pdfs if r.tcdate_ms <= limit.timestamp() * 1000]
    if before:
        return PdfChoice(
            before[-1].id, f"latest PDF revision before {deadline} + 7 days", "deadline"
        )
    return PdfChoice(
        pdfs[0].id, f"no PDF revision before {deadline} + 7 days; earliest", "earliest"
    )


def revisions_v2(body: dict) -> list[Revision]:
    return [
        Revision(
            id=e["id"],
            tcdate_ms=int(e.get("tcdate") or e.get("cdate") or 0),
            has_pdf="pdf" in ((e.get("note") or {}).get("content") or {}),
        )
        for e in body.get("edits", [])
    ]


def revisions_v1(body: dict) -> list[Revision]:
    return [
        Revision(
            id=r["id"],
            tcdate_ms=int(r.get("tcdate") or r.get("cdate") or 0),
            has_pdf=bool((r.get("content") or {}).get("pdf")),
        )
        for r in body.get("references", [])
    ]


@dataclass(frozen=True)
class PdfFetch:
    path: Path
    reason: str
    provenance: str
    """deadline_revision / earliest_revision (from history); current_deadline_dated /
    current_revised / current_undated (no history: by the PDF's CreationDate)."""


MAX_RATE_LIMIT_WAITS = 3
RETRY_RE = re.compile(r"try again in (?:(\d+) minutes?)?(?: and )?(?:(\d+) seconds?)?")
DEFAULT_WAIT_S = 20 * 60


def retry_after_s(body: str) -> int:
    """Seconds to wait from OpenReview's 429 message ('Please try again in 16 minutes and 16
    seconds'), plus a 15 s margin; 20 minutes when the message has no time."""
    m = RETRY_RE.search(body)
    if not m or not (m.group(1) or m.group(2)):
        return DEFAULT_WAIT_S
    return int(m.group(1) or 0) * 60 + int(m.group(2) or 0) + 15


CREATION_RE = re.compile(r"^CreationDate:\s+(.+)$", re.MULTILINE)


def pdf_creation_date(path: Path) -> date | None:
    """The PDF's CreationDate via pdfinfo (poppler), or None."""
    try:
        out = subprocess.run(
            ["pdfinfo", str(path)], capture_output=True, text=True, timeout=60, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = CREATION_RE.search(out)
    if not m:
        return None
    value = re.sub(r"\s+[A-Z]{2,5}$", "", m.group(1).strip())  # zone names parse only if local
    try:
        return datetime.strptime(" ".join(value.split()), "%a %b %d %H:%M:%S %Y").date()
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(value).date()
    except ValueError:
        return None


def provenance(choice: PdfChoice, created: date | None, deadline: date | None) -> str:
    """Pure: where the PDF sits relative to the version reviewers saw."""
    if choice.kind != "current":
        return f"{choice.kind}_revision"
    if created is None or deadline is None:
        return "current_undated"
    return "current_deadline_dated" if created <= deadline + REVIEW_GRACE else "current_revised"


def _atomic_write(path: Path, text: str) -> None:
    """tmp + rename, so an interrupted run never leaves a half-written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


class OpenReviewPdfClient:
    def __init__(self, settings: Settings, http: httpx.Client | None = None) -> None:
        self.settings = settings
        self.http = http or httpx.Client(
            timeout=180, headers={"User-Agent": settings.user_agent}, follow_redirects=True
        )
        self.cache = DiskCache(settings.cache_dir, "openreview", ".json")
        self._tokens: dict[str, str] = {}
        self._last = 0.0

    def close(self) -> None:
        self.http.close()

    def _throttle(self) -> None:
        wait = self.settings.openreview_delay_s - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def _token(self, api: str) -> str:
        if api not in self._tokens:
            user, pw = self.settings.openreview_username, self.settings.openreview_password
            if not user or not pw:
                raise OpenReviewError(
                    "OPENREVIEW_USERNAME / OPENREVIEW_PASSWORD not set in .env (needed to "
                    "download submission PDFs; anonymous API access is blocked)"
                )
            self._throttle()
            r = self.http.post(f"{api}/login", json={"id": user, "password": pw})
            if r.status_code != 200 or "token" not in r.json():
                raise OpenReviewError(f"OpenReview login failed: HTTP {r.status_code}")
            self._tokens[api] = str(r.json()["token"])
        return self._tokens[api]

    def _get(self, api: str, path: str, params: dict[str, str]) -> httpx.Response:
        for attempt in range(MAX_RATE_LIMIT_WAITS + 1):
            self._throttle()
            r = self.http.get(
                f"{api}{path}",
                params=params,
                headers={"Authorization": f"Bearer {self._token(api)}"},
            )
            if r.status_code == 429 and attempt < MAX_RATE_LIMIT_WAITS:
                wait = retry_after_s(r.text)
                logger.warning(
                    "OpenReview rate limit on %s; waiting %ds (%d/%d)",
                    path,
                    wait,
                    attempt + 1,
                    MAX_RATE_LIMIT_WAITS,
                )
                time.sleep(wait)
                continue
            if r.status_code != 200:
                raise OpenReviewError(f"GET {path} HTTP {r.status_code}: {r.text[:200]}")
            return r
        raise OpenReviewError(f"GET {path}: still rate-limited")  # unreachable

    def _get_json(self, api: str, path: str, params: dict[str, str]) -> dict:
        key = content_key(api, path, json.dumps(sorted(params.items())))
        if (cached := self.cache.get_json(key)) is not None:
            return cached
        body = self._get(api, path, params).json()
        self.cache.put_json(key, body)
        return body

    def revisions(self, submission_id: str, api: str) -> list[Revision]:
        if api == API_V1:
            body = self._get_json(
                api, "/references", {"referent": submission_id, "original": "true"}
            )
            return revisions_v1(body)
        return revisions_v2(
            self._get_json(api, "/notes/edits", {"note.id": submission_id, "sort": "tcdate:asc"})
        )

    def download(self, submission_id: str, venue: str, deadline: date | None) -> PdfFetch:
        """The PDF and how it was chosen. Reuses a PDF already on disk."""
        dest = self.settings.openreview_pdf_dir / f"{submission_id}.pdf"
        note = dest.with_suffix(".revision.json")
        if dest.exists() and dest.stat().st_size > 0 and note.exists():
            try:
                meta = json.loads(note.read_text(encoding="utf-8"))
                return PdfFetch(dest, meta["reason"], meta["provenance"])
            except (ValueError, KeyError):
                logger.warning("unreadable %s; downloading %s again", note.name, submission_id)
        api = api_for(venue)
        if api == API and not self.settings.openreview_v2_history:
            choice = PdfChoice(None, "API v2 edit history not visible to this account; current PDF")
        else:
            choice = choose_revision(self.revisions(submission_id, api), deadline)
        if choice.revision_id is None:
            r = self._get(api, "/pdf", {"id": submission_id})
        elif api == API_V1:
            r = self._get(api, "/references/pdf", {"id": choice.revision_id})
        else:
            r = self._get(api, "/notes/edits/attachment", {"id": choice.revision_id, "name": "pdf"})
        if not r.content.startswith(b"%PDF"):
            raise OpenReviewError(f"{submission_id}: download is not a PDF ({r.content[:40]!r})")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp")
        tmp.write_bytes(r.content)
        tmp.replace(dest)
        created = pdf_creation_date(dest)
        prov = provenance(choice, created, deadline)
        rev = choice.revision_id or "current"
        reason = f"{choice.reason} (revision {rev}; PDF created {created})"
        _atomic_write(note, json.dumps({"reason": reason, "provenance": prov}))
        logger.info("downloaded %s: %s -> %s", submission_id, reason, prov)
        return PdfFetch(dest, reason, prov)
