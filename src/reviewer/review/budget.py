"""Per-job spend limit with wait-or-raise holds (BP §4 "Per-job hard cap").

Before each uncached LLM call the caller takes a hold of the call's worst case (input tokens
estimated from the prompt + `max_tokens` output). Parallel calls whose holds would jointly
overshoot the limit wait for each other instead of failing; a call that cannot fit even alone
raises `BudgetExceeded`. Once a step's deadline has passed no new call starts (`StepTimeout`).
Cache hits never touch the budget.
"""

import logging
import math
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from reviewer.config import MODEL_PRICES_PER_MTOK

logger = logging.getLogger(__name__)

JOB_LIMIT_CAP_USD = 6.0
CHARS_PER_TOKEN_BOUND = 3
"""Offline upper bound on input tokens: ceil(chars / 3) (DECISIONS W1-4)."""


class BudgetExceeded(RuntimeError):  # noqa: N818 - the name is the plan's (BP §4)
    """A call's hold does not fit under the job limit even with nothing else in flight."""


class StepTimeout(RuntimeError):  # noqa: N818 - the name is the plan's (BP §4)
    """The step's deadline passed before this call could start."""


def estimate_input_tokens(*texts: str) -> int:
    return sum(math.ceil(len(t) / CHARS_PER_TOKEN_BOUND) for t in texts)


def hold_usd(model: str, input_tokens: int, max_tokens: int) -> float:
    """Worst-case cost of one call: all input at the input rate, `max_tokens` of output."""
    price_in, price_out = MODEL_PRICES_PER_MTOK[model]
    return (input_tokens * price_in + max_tokens * price_out) / 1e6


def job_limit(reserved_usd: float, largest_hold_usd: float) -> float:
    """min($6, max(1.5 × reserved, reserved + 2 × largest hold)) (BP §4)."""
    return min(
        JOB_LIMIT_CAP_USD,
        max(1.5 * reserved_usd, reserved_usd + 2 * largest_hold_usd),
    )


@dataclass
class Slot:
    """An admitted call. Set `actual_usd` once the call's real cost is known; if it is never
    set (the call raised without usage), the whole hold is charged: overcount, never under."""

    hold_usd: float
    actual_usd: float | None = None


class Budget:
    def __init__(
        self,
        limit_usd: float,
        clock: Callable[[], float] = time.monotonic,
        on_admit: Callable[[Slot], None] | None = None,
        on_settle: Callable[[Slot], None] | None = None,
    ) -> None:
        """`on_admit` runs after admission and before the call (the per-call ledger row goes
        here, W2); `on_settle` runs after the call with the charged amount set."""
        self.limit_usd = limit_usd
        self.spent_usd = 0.0
        self.in_flight_usd = 0.0
        self._clock = clock
        self._on_admit = on_admit
        self._on_settle = on_settle
        self._cond = threading.Condition()

    def _admit(self, hold: float, deadline: float | None) -> None:
        with self._cond:
            while True:
                if deadline is not None and self._clock() >= deadline:
                    raise StepTimeout("step deadline passed before the call started")
                if self.spent_usd + hold > self.limit_usd:
                    raise BudgetExceeded(
                        f"hold ${hold:.4f} does not fit: spent ${self.spent_usd:.4f} "
                        f"of ${self.limit_usd:.2f}"
                    )
                if self.spent_usd + self.in_flight_usd + hold <= self.limit_usd:
                    self.in_flight_usd += hold
                    return
                timeout = None if deadline is None else max(0.0, deadline - self._clock())
                self._cond.wait(timeout=timeout)

    def _settle(self, slot: Slot) -> None:
        charged = slot.hold_usd if slot.actual_usd is None else slot.actual_usd
        with self._cond:
            self.in_flight_usd -= slot.hold_usd
            self.spent_usd += charged
            self._cond.notify_all()
        slot.actual_usd = charged

    @contextmanager
    def call(self, hold: float, deadline: float | None = None) -> Iterator[Slot]:
        """Admit one uncached call. Blocks while parallel holds crowd the limit; raises
        `BudgetExceeded` or `StepTimeout` instead of starting it."""
        self._admit(hold, deadline)
        slot = Slot(hold_usd=hold)
        try:
            if self._on_admit is not None:
                self._on_admit(slot)
            yield slot
        finally:
            self._settle(slot)
            if self._on_settle is not None:
                self._on_settle(slot)
