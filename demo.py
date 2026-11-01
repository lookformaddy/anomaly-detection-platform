"""End-to-end demo on synthetic time series: fit, calibrate, stream, evaluate,
alert. Run: python demo.py (or: python -m anomaly.cli demo, from src on path)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from anomaly.alerts import AlertManager
from anomaly.calibration import percentile_threshold
from anomaly.cli import synthetic_series
from anomaly.detectors import IsolationForestDetector, MADDetector
from anomaly.evaluation import confusion, detection_delays, roc_auc
from anomaly.streaming import DriftMonitor, StreamingScorer, WindowedBaseline


def main() -> None:
    points, labels = synthetic_series(n=500, seed=7)
    train, valid = points[:150], points[150:]
    valid_labels = labels[150:]
    for name, det in [("mad", MADDetector()), ("isolation_forest", IsolationForestDetector(seed=42))]:
        det.fit(train)
        train_scores = [det.score(p) for p in train]
        threshold = percentile_threshold(train_scores, 99.0)
        scores = [det.score(p) for p in valid]
        m = confusion(scores, valid_labels, threshold)
        print(f"[{name}] threshold={threshold:.3f} precision={m.precision:.3f} "
              f"recall={m.recall:.3f} f1={m.f1:.3f} roc_auc={roc_auc(scores, valid_labels):.3f} "
              f"delays={detection_delays([int(s >= threshold) for s in scores], valid_labels)}")

    # Streaming + alerts on the full series.
    alerts = AlertManager(suppression_seconds=60.0)
    scorer = StreamingScorer(
        baseline=WindowedBaseline(window=50, robust=True),
        threshold=4.0,
        drift=DriftMonitor(reference=[0.5, 1.0, 1.5]),
    )
    paged = 0
    for p, y in zip(points, labels):
        score, is_anomaly = scorer.score_point(p)
        if is_anomaly:
            alert = alerts.raise_alert("series:demo", "streaming", p.timestamp, score, "spike detected")
            paged += alert is not None
    print(f"[streaming] pages={paged} open_alerts={len(alerts.open_alerts())} "
          f"drift_psi={scorer.drift.psi():.3f} drifted={scorer.drift.is_drifted()}")
    print("demo OK")


if __name__ == "__main__":
    main()
