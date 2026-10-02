"""Venue deadlines beta users target (MASTER_PLAN D20, D58).

Plain data, keyed `"<venue>/<kind>"`. A deadline that is not yet published is simply absent.
Adding a row is a one-line data change, deployable outside the Nov freeze.
"""

from datetime import UTC, datetime, timedelta

from reviewer.models import Deadline

DEADLINES: dict[str, Deadline] = {
    # "The submission deadline is November 16th, 2026 Anywhere on Earth (UTC-12) and will not
    # be changed." No clock time is printed; end of day AoE = Nov 17 11:59 UTC.
    "cvpr2027/paper": Deadline(
        at=datetime(2026, 11, 17, 11, 59, tzinfo=UTC),
        source_url="https://cvpr.thecvf.com/Conferences/2027/CallForPapers",
    ),
    # "Authors can also revise their paper until the end of the public discussion period
    # (Nov 18)." No time or zone is stated; assumed end of day AoE until ICLR says otherwise.
    "iclr2027/revision": Deadline(
        at=datetime(2026, 11, 19, 11, 59, tzinfo=UTC),
        source_url="https://iclr.cc/Conferences/2027/AuthorGuidelines",
        provisional=True,
    ),
}

WINDOW_BEFORE = timedelta(hours=24)
"""Skip the index update when a deadline is at most this far ahead..."""
WINDOW_AFTER = timedelta(hours=6)
"""...or at most this far behind (D58: −6 h to +24 h of now)."""


def in_deadline_window(now: datetime, rows: dict[str, Deadline]) -> tuple[str, Deadline] | None:
    """The first row whose deadline falls within [now − 6 h, now + 24 h], else None."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    for key, row in sorted(rows.items(), key=lambda kv: kv[1].at):
        if now - WINDOW_AFTER <= row.at <= now + WINDOW_BEFORE:
            return key, row
    return None
