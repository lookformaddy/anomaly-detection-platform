from anomaly.models import Point
from anomaly.streaming import DriftMonitor, SeasonalityAdjuster, StreamingScorer, WindowedBaseline


def test_windowed_baseline_scores_spike_after_warmup():
    base = WindowedBaseline(window=50)
    for i in range(50):
        base.update(100.0)
    assert base.update(200.0) > 5.0
    assert base.update(100.0) < 3.0


def test_seasonality_adjuster_removes_periodic_pattern():
    values = [10.0 if i % 2 == 0 else 20.0 for i in range(100)]
    adj = SeasonalityAdjuster(period=2).fit(values)
    assert abs(adj.adjust(0, 10.0) - adj.adjust(1, 20.0)) < 1e-9


def test_drift_monitor_detects_shift():
    mon = DriftMonitor(reference=[0.0, 0.5, 1.0, 1.5, 2.0] * 20)
    for _ in range(100):
        mon.observe(10.0)
    assert mon.is_drifted()
    assert mon.ks_statistic() == 1.0


def test_streaming_scorer_flags_spike():
    scorer = StreamingScorer(baseline=WindowedBaseline(window=50), threshold=4.0)
    for i in range(60):
        scorer.score_point(Point(timestamp=float(i), value=100.0))
    _, flagged = scorer.score_point(Point(timestamp=61.0, value=200.0))
    assert flagged
