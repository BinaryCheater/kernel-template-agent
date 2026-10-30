"""Controllable clock so tests never sleep (spec/conventions.md §9)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta


class ManualClock:
    FIXTURE = True

    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2026, 1, 1, tzinfo=UTC)

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("time does not go backwards")
        self._now += timedelta(seconds=seconds)
