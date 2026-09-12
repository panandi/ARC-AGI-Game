"""Swappable rate limiting.

The application depends only on the :class:`RateLimiter` protocol.
:class:`InMemoryRateLimiter` is a sliding-window implementation that is correct
for the single-worker MVP. For multi-worker or multi-instance deployments,
supply a Redis/Upstash-backed implementation of the same protocol via
:func:`set_rate_limiter` -- nothing else in the codebase needs to change.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    retry_after_seconds: float = 0.0


@runtime_checkable
class RateLimiter(Protocol):
    async def hit(self, key: str, limit: int, window_seconds: float) -> RateLimitDecision:
        """Record an attempt and report whether it is permitted."""
        ...


class InMemoryRateLimiter:
    """Sliding-window counter held in process memory.

    Suitable for local development and the single-worker MVP. State is lost on
    restart and is not shared across processes.
    """

    def __init__(self, max_keys: int = 10_000) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()
        self._max_keys = max_keys

    async def hit(self, key: str, limit: int, window_seconds: float) -> RateLimitDecision:
        now = time.monotonic()
        cutoff = now - window_seconds
        async with self._lock:
            if len(self._hits) > self._max_keys:
                self._evict(cutoff)
            bucket = self._hits[key]
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= limit:
                retry_after = max(0.0, bucket[0] + window_seconds - now)
                return RateLimitDecision(False, retry_after)
            bucket.append(now)
            return RateLimitDecision(True)

    def _evict(self, cutoff: float) -> None:
        stale = [k for k, v in self._hits.items() if not v or v[-1] < cutoff]
        for key in stale:
            self._hits.pop(key, None)

    def reset(self) -> None:
        """Test helper."""
        self._hits.clear()


class MinIntervalLimiter:
    """Enforces a minimum spacing between successive events for a key.

    Used to pace AI turns so a fast environment cannot spray OpenRouter calls.
    """

    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = interval_seconds
        self._last: dict[str, float] = {}

    async def wait(self, key: str) -> None:
        if self.interval_seconds <= 0:
            return
        now = time.monotonic()
        previous = self._last.get(key)
        if previous is not None:
            delay = self.interval_seconds - (now - previous)
            if delay > 0:
                await asyncio.sleep(delay)
        self._last[key] = time.monotonic()

    def forget(self, key: str) -> None:
        self._last.pop(key, None)


_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = InMemoryRateLimiter()
    return _limiter


def set_rate_limiter(limiter: RateLimiter | None) -> None:
    """Swap the implementation (production Redis backend, or a test double)."""
    global _limiter
    _limiter = limiter
