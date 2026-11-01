from anomaly.alerts import AlertManager
from anomaly.models import AlertState


def test_dedup_within_suppression_window():
    mgr = AlertManager(suppression_seconds=3600.0)
    first = mgr.raise_alert("fp", "mad", 0.0, 5.0, "spike")
    assert first is not None
    assert mgr.raise_alert("fp", "mad", 10.0, 6.0, "spike") is None
    assert len(mgr.open_alerts()) == 1
    assert mgr.open_alerts()[0].occurrences == 2


def test_acknowledge_and_resolve():
    mgr = AlertManager()
    mgr.raise_alert("fp", "mad", 0.0, 5.0)
    assert mgr.acknowledge("fp")
    assert mgr.open_alerts()[0].state == AlertState.ACKNOWLEDGED
    assert mgr.resolve("fp")
    assert mgr.open_alerts() == []
    # A new detection after resolution pages again.
    assert mgr.raise_alert("fp", "mad", 5000.0, 5.0) is not None


def test_escalation_after_repeats():
    mgr = AlertManager(suppression_seconds=0.0, escalation_after=3)
    mgr.raise_alert("fp", "mad", 0.0, 5.0)
    mgr.raise_alert("fp", "mad", 1.0, 5.0)
    escalated = mgr.raise_alert("fp", "mad", 2.0, 5.0)
    assert escalated is not None and "escalated" in escalated.message
