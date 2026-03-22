"""CacheConfig -- provider-agnostic caching intent.

Business-level knobs that let providers decide how to apply
vendor-specific caching mechanisms (breakpoints, TTLs, retention
policies) without leaking those details to callers.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class TrafficEstimator:
    """Estimate expected request volume over a time window.

    Providers call :meth:`expected_requests` with a monotonic timestamp
    and duration to decide whether caching will pay off.
    """

    def expected_requests(self, start: float, duration: float) -> float:
        """Return the expected number of requests in *[start, start+duration)*.

        Parameters
        ----------
        start : float
            ``time.monotonic()`` timestamp.
        duration : float
            Window length in seconds.
        """
        raise NotImplementedError


class ConstantRate(TrafficEstimator):
    """Assumes a fixed queries-per-second rate."""

    def __init__(self, qps: float) -> None:
        self._qps = qps

    def expected_requests(self, start: float, duration: float) -> float:
        return self._qps * duration


@dataclass
class CacheConfig:
    """Provider-agnostic caching intent.

    Parameters
    ----------
    refresh_interval : float
        Expected seconds between consecutive calls in a single session.
        Providers may use this to decide TTL tiers.
    traffic : TrafficEstimator
        Estimates request volume so providers can compare cache write
        cost vs. read savings across TTL options.
    """

    refresh_interval: float = 30.0
    traffic: TrafficEstimator = field(
        default_factory=lambda: ConstantRate(qps=0)
    )
