import pytest

from anomaly.evaluation import confusion, detection_delays, roc_auc


def test_perfect_scores():
    scores = [0.1, 0.2, 0.9, 0.95]
    labels = [0, 0, 1, 1]
    m = confusion(scores, labels, 0.5)
    assert (m.precision, m.recall, m.f1) == (1.0, 1.0, 1.0)
    assert roc_auc(scores, labels) == 1.0


def test_chance_auc():
    assert roc_auc([0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]) == 0.5


def test_detection_delays_and_miss():
    labels = [0, 1, 1, 1, 0, 1, 1]
    preds = [0, 0, 1, 0, 0, 0, 0]
    assert detection_delays(preds, labels) == [1, None]
    with pytest.raises(ValueError):
        detection_delays([0, 1], labels)
