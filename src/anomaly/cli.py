"""CLI: fit a detector on synthetic (or CSV) data and score a stream.

Usage:
    python -m anomaly.cli demo
    python -m anomaly.cli score --detector mad --threshold 3.5 --points 100
"""

from __future__ import annotations

import argparse
import random

from anomaly.calibration import percentile_threshold
from anomaly.detectors import MADDetector, ZScoreDetector
from anomaly.evaluation import confusion, roc_auc
from anomaly.models import Point
from anomaly.streaming import StreamingScorer, WindowedBaseline


def synthetic_series(n: int = 500, seed: int = 7) -> tuple[list[Point], list[int]]:
    rng = random.Random(seed)
    points, labels = [], []
    for i in range(n):
        value = 100.0 + 10.0 * rng.gauss(0, 1)
        label = 0
        if 200 <= i < 210 or 400 <= i < 405:  # injected level shifts
            value += 60.0
            label = 1
        points.append(Point(timestamp=float(i), value=value))
        labels.append(label)
    return points, labels


def run_demo() -> dict:
    points, labels = synthetic_series()
    train, valid = points[:150], points[150:]
    valid_labels = labels[150:]
    det = MADDetector().fit(train)
    scores = [det.score(p) for p in valid]
    threshold = percentile_threshold([det.score(p) for p in train], 99.0)
    metrics = confusion(scores, valid_labels, threshold)
    auc = roc_auc(scores, valid_labels)
    return {
        "threshold": round(threshold, 3),
        "precision": round(metrics.precision, 3),
        "recall": round(metrics.recall, 3),
        "f1": round(metrics.f1, 3),
        "roc_auc": round(auc, 3),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="anomaly.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo")
    score_p = sub.add_parser("score")
    score_p.add_argument("--detector", choices=["mad", "zscore"], default="mad")
    score_p.add_argument("--threshold", type=float, default=3.5)
    score_p.add_argument("--points", type=int, default=100)
    args = parser.parse_args(argv)
    if args.command == "demo":
        for k, v in run_demo().items():
            print(f"{k}: {v}")
        return 0
    cls = MADDetector if args.detector == "mad" else ZScoreDetector
    points, _ = synthetic_series(n=args.points)
    det = cls().fit(points[: max(10, args.points // 3)])
    scorer = StreamingScorer(baseline=WindowedBaseline(window=50), threshold=args.threshold)
    flagged = sum(1 for p in points if scorer.score_point(p)[1])
    print(f"detector={args.detector} flagged={flagged}/{len(points)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
