"""Streaming scoring: windowed baselines, seasonality adjustment, and drift
monitoring of score distributions. The batch detectors assume a stationary
training set; production series drift, so the streaming layer re-baselines
continuously instead of trusting a stale fit."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from anomaly.detectors import _mean, _median, _std
from anomaly.models import Point


@dataclass
class WindowedBaseline:
    """Rolling mean/std (or median/MAD) over the last ``window`` points.
    A point is scored against the window *before* it, so an anomaly cannot
    immediately normalise itself."""

    window: int = 100
    robust: bool = False
    _values: deque = field(default_factory=deque, repr=False)

    def update(self, value: float) -> float:
        """Score ``value`` against the current window, then ingest it."""
        if len(self._values) < 10:
            score = 0.0
        elif self.robust:
            med = _median(list(self._values))
            mad = _median([abs(v - med) for v in self._values]) or 1.0
            score = 0.6745 * abs(value - med) / mad
        else:
            mean = _mean(list(self._values))
            score = abs(value - mean) / _std(list(self._values), mean)
        self._values.append(value)
        while len(self._values) > self.window:
            self._values.popleft()
        return score


@dataclass
class SeasonalityAdjuster:
    """Removes a repeating seasonal profile (median per bucket) so daily or
    weekly patterns are not flagged as anomalies. Profile is learned from a
    warm-up pass; unseen buckets fall back to the global median."""

    period: int = 24
    profile_: dict[int, float] = field(default_factory=dict)
    global_median_: float = 0.0

    def fit(self, values: list[float]) -> "SeasonalityAdjuster":
        self.global_median_ = _median(values)
        buckets: dict[int, list[float]] = {}
        for i, v in enumerate(values):
            buckets.setdefault(i % self.period, []).append(v)
        self.profile_ = {b: _median(vs) for b, vs in buckets.items()}
        return self

    def adjust(self, index: int, value: float) -> float:
        seasonal = self.profile_.get(index % self.period, self.global_median_)
        return value - seasonal + self.global_median_


@dataclass
class DriftMonitor:
    """Compares a reference score distribution with a recent window using PSI
    (population stability index) and the KS statistic, both from scratch.
    PSI > 0.25 is the conventional 'significant shift' band; KS is reported
    alongside because PSI depends on binning."""

    reference: list[float] = field(default_factory=list)
    window: int = 200
    _recent: deque = field(default_factory=deque, repr=False)

    def observe(self, score: float) -> None:
        self._recent.append(score)
        while len(self._recent) > self.window:
            self._recent.popleft()

    def psi(self, n_bins: int = 10) -> float:
        if not self.reference or not self._recent:
            return 0.0
        ref = sorted(self.reference)
        edges = [ref[min(int(i * len(ref) / n_bins), len(ref) - 1)] for i in range(n_bins + 1)]
        edges[0], edges[-1] = -math.inf, math.inf
        eps = 1e-4
        total = 0.0
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            a = sum(1 for x in self.reference if lo <= x < hi or (i == n_bins - 1 and x == hi)) / len(self.reference)
            b = sum(1 for x in self._recent if lo <= x < hi or (i == n_bins - 1 and x == hi)) / len(self._recent)
            a, b = max(a, eps), max(b, eps)
            total += (a - b) * math.log(a / b)
        return total

    def ks_statistic(self) -> float:
        if not self.reference or not self._recent:
            return 0.0
        combined = sorted(set(self.reference) | set(self._recent))
        worst = 0.0
        for x in combined:
            cdf_r = sum(1 for v in self.reference if v <= x) / len(self.reference)
            cdf_w = sum(1 for v in self._recent if v <= x) / len(self._recent)
            worst = max(worst, abs(cdf_r - cdf_w))
        return worst

    def is_drifted(self, psi_threshold: float = 0.25) -> bool:
        return self.psi() > psi_threshold


@dataclass
class StreamingScorer:
    """Online scorer: seasonality-adjust, then score against a rolling
    windowed baseline. ``threshold`` is calibrated offline (see calibration)."""

    baseline: WindowedBaseline = field(default_factory=WindowedBaseline)
    seasonality: SeasonalityAdjuster | None = None
    threshold: float = 3.5
    drift: DriftMonitor | None = None
    _index: int = 0

    def score_point(self, point: Point) -> tuple[float, bool]:
        value = point.value
        if self.seasonality is not None:
            value = self.seasonality.adjust(self._index, value)
        score = self.baseline.update(value)
        if self.drift is not None:
            self.drift.observe(score)
        self._index += 1
        return score, score >= self.threshold
