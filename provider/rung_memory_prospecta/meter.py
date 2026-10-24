"""Per-hook call and cost counting. A hook runs on one worker thread, so a thread-local suffices."""

from __future__ import annotations

import threading


class BudgetExceeded(Exception):
    """The next paid call would take the hook past its cost budget."""


class Meter:
    def __init__(self) -> None:
        self._local = threading.local()

    def reset(self, limit: float | None = None) -> None:
        """Start a hook. `limit` is the most it may spend (USD); None means unbounded."""
        self._local.calls = 0
        self._local.cost = 0.0
        self._local.limit = limit

    def check(self, estimate: float = 0.0) -> None:
        """Raise BudgetExceeded if a call expected to cost `estimate` would pass the hook's limit."""
        limit = getattr(self._local, "limit", None)
        if limit is None:
            return
        if limit <= 0 or getattr(self._local, "cost", 0.0) + estimate > limit:
            raise BudgetExceeded(f"cost budget {limit:g} USD would be exceeded")

    def add(self, cost: float) -> None:
        self._local.calls = getattr(self._local, "calls", 0) + 1
        self._local.cost = getattr(self._local, "cost", 0.0) + float(cost or 0.0)

    @property
    def calls(self) -> int:
        return getattr(self._local, "calls", 0)

    @property
    def cost(self) -> float:
        return round(getattr(self._local, "cost", 0.0), 6)
