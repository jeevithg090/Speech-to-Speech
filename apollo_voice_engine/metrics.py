"""Small helpers for latency/request metrics reported by the demo servers."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any


class RunningMetrics:
    """Thread-safe counters and running averages keyed by name."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {}
        self._averages: dict[str, float] = {}
        self._samples: dict[str, int] = {}
        self.started_at = datetime.now(timezone.utc).isoformat()

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + amount

    def observe(self, name: str, value: float) -> None:
        """Fold ``value`` into the running mean for ``name``."""
        with self._lock:
            n = self._samples.get(name, 0) + 1
            self._samples[name] = n
            prev = self._averages.get(name, 0.0)
            self._averages[name] = prev + (value - prev) / n

    def counter(self, name: str) -> int:
        with self._lock:
            return self._counters.get(name, 0)

    def average(self, name: str) -> float | None:
        with self._lock:
            return self._averages.get(name)

    def snapshot(self) -> dict[str, Any]:
        """Return counters plus ``avg_<name>`` entries (rounded to 0.1)."""
        with self._lock:
            data: dict[str, Any] = dict(self._counters)
            data.update({f"avg_{k}": round(v, 1) for k, v in self._averages.items()})
        data["started_at"] = self.started_at
        return data
