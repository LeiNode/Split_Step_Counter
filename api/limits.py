"""
Per-client rate limiting for the public API.

In-memory sliding window per client IP. Behind a Lambda Function URL the IP
comes from the request context (set by Mangum), not from a header a caller
could forge. Each Lambda container keeps its own window, so the effective
limit is RATE_LIMIT_PER_HOUR x the number of warm containers (capped by the
function's reserved concurrency).

Running uvicorn behind your own reverse proxy? Start it with
--proxy-headers --forwarded-allow-ips=<proxy ip> so request.client is the
real caller, not the proxy.
"""

import os
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

WINDOW_S = 3600


class RateLimiter:
    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, limit: int) -> None:
        """Record a hit for key; raise 429 if it would exceed limit per WINDOW_S."""
        now = self._clock()
        with self._lock:
            q = self._hits[key]
            while q and q[0] <= now - WINDOW_S:
                q.popleft()
            if len(q) >= limit:
                retry = int(q[0] + WINDOW_S - now) + 1
                raise HTTPException(status_code=429, headers={"Retry-After": str(retry)},
                                    detail=f"Rate limit: {limit} analyses per hour. Try again in {retry // 60 + 1} min.")
            q.append(now)
            if len(self._hits) > 10_000:          # forget idle clients so memory stays bounded
                for k in [k for k, v in self._hits.items() if not v]:
                    del self._hits[k]


_limiter = RateLimiter()


def rate_limit(request: Request) -> None:
    """Dependency: enforce RATE_LIMIT_PER_HOUR (unset or 0 = no limit) per client IP."""
    limit = int(os.environ.get("RATE_LIMIT_PER_HOUR", "0") or 0)
    if limit > 0:
        _limiter.check(request.client.host if request.client else "unknown", limit)
