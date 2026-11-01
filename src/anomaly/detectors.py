"""Detectors built from scratch (no sklearn): z-score, robust MAD,
isolation-forest-style random-partition trees, and one-class centroid distance.

All detectors share one contract: ``fit(points)`` then ``score(point)`` in
[0, 1]-ish unbounded positive space where larger means more anomalous, plus
``predict(point, threshold)``. Randomness is always seeded for reproducibility.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from anomaly.models import Point


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float], mean: float) -> float:
    if len(xs) < 2:
        return 1.0
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var) or 1.0


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


class BaseDetector:
    name = "base"

    def fit(self, points: list[Point]) -> "BaseDetector":
        raise NotImplementedError

    def score(self, point: Point) -> float:
        raise NotImplementedError

    def predict(self, point: Point, threshold: float) -> bool:
        return self.score(point) >= threshold


@dataclass
class ZScoreDetector(BaseDetector):
    name: str = "zscore"
    mean_: float = 0.0
    std_: float = 1.0

    def fit(self, points: list[Point]) -> "ZScoreDetector":
        vals = [p.value for p in points]
        self.mean_ = _mean(vals)
        self.std_ = _std(vals, self.mean_)
        return self

    def score(self, point: Point) -> float:
        return abs(point.value - self.mean_) / self.std_


@dataclass
class MADDetector(BaseDetector):
    """Robust alternative to z-score: median +/- scaled MAD, insensitive to
    the very outliers it must detect (z-score's mean/std are pulled by them)."""

    name: str = "mad"
    median_: float = 0.0
    mad_: float = 1.0

    def fit(self, points: list[Point]) -> "MADDetector":
        vals = [p.value for p in points]
        self.median_ = _median(vals)
        self.mad_ = _median([abs(v - self.median_) for v in vals]) or 1.0
        return self

    def score(self, point: Point) -> float:
        # 0.6745 scales MAD to be comparable to a standard deviation.
        return 0.6745 * abs(point.value - self.median_) / self.mad_


@dataclass
class _IsoNode:
    feature: int = -1
    split: float = 0.0
    left: "_IsoNode | None" = None
    right: "_IsoNode | None" = None
    size: int = 0


def _c_factor(n: int) -> float:
    """Average path length of an unsuccessful search in a BST of n nodes."""
    if n <= 1:
        return 0.0
    harmonic = sum(1.0 / i for i in range(1, n))
    return 2.0 * harmonic - 2.0 * (n - 1) / n


def _path_length(node: _IsoNode, vec: list[float], depth: int = 0) -> float:
    if node.left is None or node.right is None:
        return depth + _c_factor(node.size)
    if vec[node.feature] < node.split:
        return _path_length(node.left, vec, depth + 1)
    return _path_length(node.right, vec, depth + 1)


@dataclass
class IsolationForestDetector(BaseDetector):
    """Isolation-forest-style detector: anomalies are isolated in few random
    partitions, so their average path length is short. Score is
    2**(-E[path] / c(n)), in (0, 1); ~0.5 is normal, near 1 is anomalous."""

    name: str = "isolation_forest"
    n_trees: int = 100
    sample_size: int = 256
    seed: int = 42
    trees: list[_IsoNode] = field(default_factory=list)
    sample_size_: int = 0

    def _build(self, rows: list[list[float]], rng: random.Random, depth: int, max_depth: int) -> _IsoNode:
        node = _IsoNode(size=len(rows))
        if len(rows) <= 1 or depth >= max_depth:
            return node
        n_feat = len(rows[0])
        feat = rng.randrange(n_feat)
        vals = [r[feat] for r in rows]
        lo, hi = min(vals), max(vals)
        if lo == hi:
            return node
        node.feature, node.split = feat, rng.uniform(lo, hi)
        left = [r for r in rows if r[feat] < node.split]
        right = [r for r in rows if r[feat] >= node.split]
        if not left or not right:
            return _IsoNode(size=len(rows))
        node.left = self._build(left, rng, depth + 1, max_depth)
        node.right = self._build(right, rng, depth + 1, max_depth)
        return node

    def fit(self, points: list[Point]) -> "IsolationForestDetector":
        rows = [p.vector() for p in points]
        rng = random.Random(self.seed)
        self.sample_size_ = min(self.sample_size, len(rows))
        max_depth = math.ceil(math.log2(self.sample_size_)) if self.sample_size_ > 1 else 0
        self.trees = []
        for _ in range(self.n_trees):
            sample = rng.sample(rows, self.sample_size_) if len(rows) > self.sample_size_ else list(rows)
            self.trees.append(self._build(sample, rng, 0, max_depth))
        return self

    def score(self, point: Point) -> float:
        if not self.trees:
            raise RuntimeError("detector not fitted")
        vec = point.vector()
        avg = sum(_path_length(t, vec) for t in self.trees) / len(self.trees)
        c = _c_factor(self.sample_size_)
        return 2.0 ** (-avg / c) if c else 0.5


@dataclass
class CentroidDetector(BaseDetector):
    """One-class detector: distance from the normal centroid, normalised by
    per-dimension scale. Simple, explainable, and a good multivariate baseline
    before reaching for the forest."""

    name: str = "centroid"
    centroid_: list[float] = field(default_factory=list)
    scale_: list[float] = field(default_factory=list)

    def fit(self, points: list[Point]) -> "CentroidDetector":
        rows = [p.vector() for p in points]
        dims = len(rows[0]) if rows else 0
        self.centroid_ = [_mean([r[d] for r in rows]) for d in range(dims)]
        self.scale_ = [
            _std([r[d] for r in rows], self.centroid_[d]) for d in range(dims)
        ]
        return self

    def score(self, point: Point) -> float:
        vec = point.vector()
        if len(vec) != len(self.centroid_):
            raise ValueError("feature dimension mismatch")
        return math.sqrt(sum(((v - c) / s) ** 2 for v, c, s in zip(vec, self.centroid_, self.scale_)))
