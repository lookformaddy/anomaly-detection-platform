"""Shared value types. Pure stdlib, no I/O, no network."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class Point:
    """One observation. ``features`` is empty for univariate series."""

    timestamp: float
    value: float
    features: tuple[float, ...] = ()

    def vector(self) -> list[float]:
        return [self.value, *self.features]


@dataclass(frozen=True)
class Detection:
    point: Point
    score: float
    is_anomaly: bool
    detector: str
    threshold: float


@dataclass
class DetectorConfig:
    name: str
    threshold: float = 3.0
    seed: int = 42
    params: dict[str, Any] = field(default_factory=dict)


class AlertState(str, Enum):
    NEW = "new"
    ACKNOWLEDGED = "acknowledged"
    SUPPRESSED = "suppressed"
    RESOLVED = "resolved"


@dataclass
class Alert:
    alert_id: str
    fingerprint: str
    detector: str
    state: AlertState
    first_seen: float
    last_seen: float
    occurrences: int = 1
    peak_score: float = 0.0
    message: str = ""
