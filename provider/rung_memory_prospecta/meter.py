"""Per-hook call and cost counting. A hook runs on one worker thread, so a thread-local suffices."""

from __future__ import annotations

import threading


class Meter:
    def __init__(self) -> None:
        self._local = threading.local()

    def reset(self) -> None:
        self._local.calls = 0
        self._local.cost = 0.0

    def add(self, cost: float) -> None:
        self._local.calls = getattr(self._local, "calls", 0) + 1
        self._local.cost = getattr(self._local, "cost", 0.0) + float(cost or 0.0)

    @property
    def calls(self) -> int:
        return getattr(self._local, "calls", 0)

    @property
    def cost(self) -> float:
        return round(getattr(self._local, "cost", 0.0), 6)
