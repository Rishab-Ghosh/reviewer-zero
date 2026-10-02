import threading
import time

import pytest

from reviewer.review.budget import (
    Budget,
    BudgetExceeded,
    StepTimeout,
    estimate_input_tokens,
    hold_usd,
    job_limit,
)


def _run_parallel(budget: Budget, n: int, hold: float, actual: float) -> tuple[int, list]:
    active = 0
    peak = 0
    lock = threading.Lock()
    errors: list[BaseException] = []

    def worker() -> None:
        nonlocal active, peak
        try:
            with budget.call(hold) as slot:
                with lock:
                    active += 1
                    peak = max(peak, active)
                time.sleep(0.02)
                with lock:
                    active -= 1
                slot.actual_usd = actual
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    return peak, errors


def test_six_parallel_holds_serialize_under_a_tight_limit():
    # Two $0.50 holds never fit together under $0.90, but each fits alone after 5 x $0.05.
    budget = Budget(limit_usd=0.9)
    peak, errors = _run_parallel(budget, n=6, hold=0.5, actual=0.05)
    assert errors == []
    assert peak == 1
    assert budget.spent_usd == pytest.approx(0.30)
    assert budget.in_flight_usd == pytest.approx(0.0)


def test_parallel_calls_run_together_when_they_fit():
    budget = Budget(limit_usd=10.0)
    peak, errors = _run_parallel(budget, n=6, hold=0.5, actual=0.1)
    assert errors == [] and peak > 1


def test_oversize_hold_raises():
    budget = Budget(limit_usd=0.7)
    with pytest.raises(BudgetExceeded), budget.call(0.8):
        pass
    assert budget.spent_usd == 0.0


def test_raises_once_spend_leaves_no_room():
    budget = Budget(limit_usd=1.0)
    with budget.call(0.5) as slot:
        slot.actual_usd = 0.6
    with pytest.raises(BudgetExceeded), budget.call(0.5):
        pass


def test_unset_actual_charges_the_hold():
    budget = Budget(limit_usd=1.0)
    with pytest.raises(RuntimeError, match="api"), budget.call(0.4):
        raise RuntimeError("api down")
    assert budget.spent_usd == pytest.approx(0.4)
    assert budget.in_flight_usd == pytest.approx(0.0)


def test_no_call_starts_after_the_deadline():
    now = [100.0]
    budget = Budget(limit_usd=1.0, clock=lambda: now[0])
    with budget.call(0.1, deadline=101.0):
        pass
    now[0] = 101.0
    with pytest.raises(StepTimeout), budget.call(0.1, deadline=101.0):
        pass


def test_waiting_call_times_out_at_the_deadline():
    budget = Budget(limit_usd=1.0)
    started = threading.Event()
    release = threading.Event()

    def holder() -> None:
        with budget.call(0.9) as slot:
            started.set()
            release.wait(2)
            slot.actual_usd = 0.1

    t = threading.Thread(target=holder)
    t.start()
    started.wait(2)
    with pytest.raises(StepTimeout), budget.call(0.5, deadline=time.monotonic() + 0.05):
        pass
    release.set()
    t.join(2)


def test_ledger_hooks_run_in_order():
    events: list[tuple[str, float, float | None]] = []
    budget = Budget(
        limit_usd=1.0,
        on_admit=lambda s: events.append(("admit", s.hold_usd, s.actual_usd)),
        on_settle=lambda s: events.append(("settle", s.hold_usd, s.actual_usd)),
    )
    with budget.call(0.3) as slot:
        events.append(("call", slot.hold_usd, None))
        slot.actual_usd = 0.12
    assert events == [("admit", 0.3, None), ("call", 0.3, None), ("settle", 0.3, 0.12)]


def test_hold_and_limit_arithmetic():
    assert estimate_input_tokens("abcdef", "a") == 3
    # Opus 5.5: $4 in, $20 out per Mtok.
    assert hold_usd("claude-opus-5-5", 10_000, 12_000) == pytest.approx(0.04 + 0.24)
    # BP §4 example: $0.30 reserved, $0.35 largest hold -> $1.00.
    assert job_limit(0.30, 0.35) == pytest.approx(1.00)
    assert job_limit(3.64, 0.35) == pytest.approx(5.46)
    assert job_limit(10.0, 0.5) == 6.0
