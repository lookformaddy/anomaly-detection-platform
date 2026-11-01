"""Threshold calibration. A detector score is useless without a threshold, and
a threshold picked on training scores is optimistic -- calibrate on a held-out
validation set with labelled anomalies where possible."""

from __future__ import annotations

import statistics


def percentile_threshold(scores: list[float], percentile: float) -> float:
    """Threshold at the given percentile of (normal) scores, e.g. 99.0."""
    if not scores:
        raise ValueError("no scores")
    if not 0.0 < percentile < 100.0:
        raise ValueError("percentile must be in (0, 100)")
    return float(statistics.quantiles(scores, n=100, method="inclusive")[int(percentile) - 1]) if percentile >= 1 else min(scores)


def precision_targeted_threshold(
    scores: list[float], labels: list[int], target_precision: float
) -> float:
    """Highest-recall threshold whose precision is at least ``target_precision``.

    Sweeps candidate thresholds (each unique score) in descending order; among
    thresholds meeting the precision target, picks the one flagging the most
    true positives. Falls back to the max score if no threshold qualifies.
    """
    if len(scores) != len(labels):
        raise ValueError("scores/labels length mismatch")
    if not scores:
        raise ValueError("no scores")
    best: float | None = None
    best_tp = -1
    for cand in sorted(set(scores), reverse=True):
        tp = sum(1 for s, y in zip(scores, labels) if s >= cand and y == 1)
        fp = sum(1 for s, y in zip(scores, labels) if s >= cand and y == 0)
        if tp + fp == 0:
            continue
        precision = tp / (tp + fp)
        if precision >= target_precision and tp > best_tp:
            best, best_tp = cand, tp
    return float(best) if best is not None else float(max(scores))


def fpr_targeted_threshold(scores: list[float], labels: list[int], target_fpr: float) -> float:
    """Threshold whose false-positive rate on negatives is at most ``target_fpr``,
    chosen to maximise true positives (i.e. the lowest such threshold)."""
    if len(scores) != len(labels):
        raise ValueError("scores/labels length mismatch")
    n_neg = sum(1 for y in labels if y == 0)
    if n_neg == 0 or not scores:
        raise ValueError("need negative examples and scores")
    best: float | None = None
    for cand in sorted(set(scores)):
        fp = sum(1 for s, y in zip(scores, labels) if s >= cand and y == 0)
        if fp / n_neg <= target_fpr:
            best = cand
            break
    return float(best) if best is not None else float(max(scores) + 1.0)
