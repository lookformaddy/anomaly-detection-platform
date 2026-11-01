"""Offline evaluation, all from scratch: precision/recall/F1, ROC-AUC via
rank statistics, and detection delay per anomalous segment."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassificationMetrics:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def confusion(scores: list[float], labels: list[int], threshold: float) -> ClassificationMetrics:
    if len(scores) != len(labels):
        raise ValueError("scores/labels length mismatch")
    tp = fp = tn = fn = 0
    for s, y in zip(scores, labels):
        pred = 1 if s >= threshold else 0
        if pred and y:
            tp += 1
        elif pred and not y:
            fp += 1
        elif not pred and not y:
            tn += 1
        else:
            fn += 1
    return ClassificationMetrics(tp, fp, tn, fn)


def roc_auc(scores: list[float], labels: list[int]) -> float:
    """Mann-Whitney rank AUC with average ranks for ties. 0.5 = chance."""
    if len(scores) != len(labels):
        raise ValueError("scores/labels length mismatch")
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("need both classes")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    rank_sum_pos = sum(r for r, y in zip(ranks, labels) if y == 1)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def detection_delays(predictions: list[int], labels: list[int]) -> list[int | None]:
    """Per contiguous anomalous segment: steps from segment start to the first
    flagged point, or None if the segment was missed entirely. Delay, not just
    accuracy, is what operators feel -- a detector that flags an outage an
    hour late is barely better than none."""
    if len(predictions) != len(labels):
        raise ValueError("predictions/labels length mismatch")
    delays: list[int | None] = []
    i = 0
    while i < len(labels):
        if labels[i] == 1:
            start = i
            delay: int | None = None
            while i < len(labels) and labels[i] == 1:
                if delay is None and predictions[i] == 1:
                    delay = i - start
                i += 1
            delays.append(delay)
        else:
            i += 1
    return delays
