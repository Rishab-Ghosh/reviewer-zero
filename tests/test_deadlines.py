from datetime import UTC, datetime, timedelta

import httpx
from typer.testing import CliRunner

from reviewer.deadlines import DEADLINES, in_deadline_window
from reviewer.web.cli import EXIT_RUN, EXIT_SKIP, app, guard_decision


def _utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=UTC)


def test_real_table_has_the_beta_rows():
    assert {"cvpr2027/paper", "iclr2027/revision"} <= DEADLINES.keys()
    for row in DEADLINES.values():
        assert row.at.tzinfo is not None
        assert row.source_url.startswith("https://")


def test_cvpr_deadline_is_end_of_nov16_aoe():
    assert DEADLINES["cvpr2027/paper"].at == _utc(2026, 11, 17, 11, 59)


def test_in_window_on_cvpr_deadline_morning_real_table():
    # 09:00 UTC on Nov 17 is 3 h before the CVPR deadline: the update must skip.
    hit = in_deadline_window(_utc(2026, 11, 17, 9), DEADLINES)
    assert hit is not None and hit[0] == "cvpr2027/paper"


def test_in_window_on_iclr_revision_morning_real_table():
    hit = in_deadline_window(_utc(2026, 11, 19, 9), DEADLINES)
    assert hit is not None and hit[0] == "iclr2027/revision"


def test_not_in_window_nov20_real_table():
    assert in_deadline_window(_utc(2026, 11, 20, 9), DEADLINES) is None


def test_window_edges():
    at = DEADLINES["cvpr2027/paper"].at
    rows = {"cvpr2027/paper": DEADLINES["cvpr2027/paper"]}

    assert in_deadline_window(at - timedelta(hours=24), rows) is not None
    assert in_deadline_window(at - timedelta(hours=24, minutes=1), rows) is None
    assert in_deadline_window(at + timedelta(hours=6), rows) is not None
    assert in_deadline_window(at + timedelta(hours=6, minutes=1), rows) is None


def test_guard_exit_codes_and_ping_on_skip():
    pings: list[tuple[str, str]] = []
    code = guard_decision(
        _utc(2026, 11, 17, 9), DEADLINES, "https://hc.example/x", lambda u, b: pings.append((u, b))
    )
    assert code == EXIT_SKIP
    assert pings and pings[0][1].startswith("skipped: cvpr2027/paper")
    pings.clear()
    code = guard_decision(
        _utc(2026, 11, 20, 9), DEADLINES, "https://hc.example/x", lambda u, b: pings.append((u, b))
    )
    assert code == EXIT_RUN and pings == []


def test_guard_ping_failure_still_skips():
    def boom(url: str, body: str) -> None:
        raise httpx.ConnectError("down")

    assert guard_decision(_utc(2026, 11, 17, 9), DEADLINES, "https://hc.example/x", boom) == 1


def test_cli_exit_codes(monkeypatch):
    monkeypatch.setattr("reviewer.web.cli._ping", lambda u, b: None)
    runner = CliRunner()
    assert runner.invoke(app, ["deadline-guard", "--now", "2026-11-17T09:00+00:00"]).exit_code == 1
    assert runner.invoke(app, ["deadline-guard", "--now", "2026-11-20T09:00+00:00"]).exit_code == 0
    assert runner.invoke(app, ["deadline-guard", "--now", "2026-11-20T09:00"]).exit_code == 2
