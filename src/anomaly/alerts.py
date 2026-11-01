"""Alert lifecycle: dedup by fingerprint, suppression windows, acknowledgement,
escalation on repeats, and resolution. Alerting is where a good detector
becomes a usable product -- or a pager-fatigue machine."""

from __future__ import annotations

from dataclasses import dataclass, field

from anomaly.models import Alert, AlertState


@dataclass
class AlertManager:
    """One alert per fingerprint while it is open. Repeat detections inside
    ``suppression_seconds`` increment the occurrence count without paging
    again; after ``escalation_after`` repeats, the alert is re-raised as a new
    page (a genuinely persistent anomaly deserves a second look)."""

    suppression_seconds: float = 3600.0
    escalation_after: int = 5
    _open: dict[str, Alert] = field(default_factory=dict, repr=False)
    _counter: int = 0

    def _new_alert(self, fingerprint: str, detector: str, now: float, score: float, message: str) -> Alert:
        self._counter += 1
        alert = Alert(
            alert_id=f"AL-{self._counter:06d}",
            fingerprint=fingerprint,
            detector=detector,
            state=AlertState.NEW,
            first_seen=now,
            last_seen=now,
            peak_score=score,
            message=message,
        )
        self._open[fingerprint] = alert
        return alert

    def raise_alert(self, fingerprint: str, detector: str, now: float, score: float, message: str = "") -> Alert | None:
        """Return the alert to page, or None if suppressed (deduped)."""
        existing = self._open.get(fingerprint)
        if existing is None or existing.state == AlertState.RESOLVED:
            return self._new_alert(fingerprint, detector, now, score, message)
        existing.occurrences += 1
        existing.last_seen = now
        existing.peak_score = max(existing.peak_score, score)
        if now - existing.first_seen < self.suppression_seconds:
            return None
        if existing.occurrences >= self.escalation_after:
            # Escalation: close the old alert, page a fresh one.
            existing.state = AlertState.RESOLVED
            return self._new_alert(fingerprint, detector, now, score, message + " [escalated]")
        return None

    def acknowledge(self, fingerprint: str) -> bool:
        alert = self._open.get(fingerprint)
        if alert is None or alert.state == AlertState.RESOLVED:
            return False
        alert.state = AlertState.ACKNOWLEDGED
        return True

    def resolve(self, fingerprint: str) -> bool:
        alert = self._open.get(fingerprint)
        if alert is None:
            return False
        alert.state = AlertState.RESOLVED
        return True

    def open_alerts(self) -> list[Alert]:
        return [a for a in self._open.values() if a.state != AlertState.RESOLVED]
