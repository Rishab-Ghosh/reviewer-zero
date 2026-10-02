"""`reviewer-web`: server-side admin commands (MASTER_PLAN §4.7). W1 has only the D58 guard."""

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Annotated

import httpx
import typer

from reviewer.config import get_settings
from reviewer.deadlines import DEADLINES, in_deadline_window
from reviewer.logging_setup import configure_logging
from reviewer.models import Deadline

logger = logging.getLogger(__name__)

app = typer.Typer(no_args_is_help=True, help="Reviewer Zero server admin commands.")

EXIT_RUN = 0
EXIT_SKIP = 1
"""systemd `ExecCondition=`: exit 1–254 skips the unit without marking it failed."""


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    configure_logging(verbose)


def _ping(url: str, body: str) -> None:
    httpx.post(url, content=body.encode(), timeout=10.0)


def guard_decision(
    now: datetime,
    rows: dict[str, Deadline],
    ping_url: str | None,
    ping: Callable[[str, str], None] = _ping,
) -> int:
    """Exit code for the nightly index update: EXIT_SKIP inside a deadline window, else
    EXIT_RUN. On a skip it pings healthchecks (the update itself pings when it runs)."""
    hit = in_deadline_window(now, rows)
    if hit is None:
        logger.info("deadline-guard: no deadline within the window; the update runs")
        return EXIT_RUN
    key, row = hit
    msg = f"skipped: {key} at {row.at.isoformat()}" + (" (provisional)" if row.provisional else "")
    logger.info("deadline-guard: %s", msg)
    if ping_url:
        try:
            ping(ping_url, msg)
        except httpx.HTTPError as e:
            logger.warning("deadline-guard: healthchecks ping failed: %s", type(e).__name__)
    return EXIT_SKIP


@app.command("deadline-guard")
def deadline_guard(
    now: Annotated[
        str | None, typer.Option(help="ISO 8601 instant to evaluate instead of the clock (tests)")
    ] = None,
    show: Annotated[bool, typer.Option(help="Print the table this guard acts on")] = False,
) -> None:
    """Exit 1 (skip) if a DEADLINES row is within −6 h/+24 h of now, else 0 (D58)."""
    when = datetime.fromisoformat(now) if now else datetime.now(UTC)
    if when.tzinfo is None:
        raise typer.BadParameter("--now needs a timezone, e.g. 2026-11-17T09:00+00:00")
    if show:
        for key, row in sorted(DEADLINES.items(), key=lambda kv: kv[1].at):
            flag = " provisional" if row.provisional else ""
            typer.echo(f"{key}\t{row.at.isoformat()}{flag}\t{row.source_url}")
    code = guard_decision(when, DEADLINES, get_settings().healthchecks_index_update_url)
    raise typer.Exit(code)
