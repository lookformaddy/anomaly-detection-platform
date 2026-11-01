import pytest

from anomaly.calibration import fpr_targeted_threshold, percentile_threshold, precision_targeted_threshold


def test_percentile_threshold():
    scores = list(range(1, 101))
    assert percentile_threshold(scores, 99.0) == pytest.approx(99.0, abs=1.0)
    with pytest.raises(ValueError):
        percentile_threshold([], 99.0)


def test_precision_targeted_threshold():
    scores = [0.1, 0.2, 0.9, 0.95]
    labels = [0, 0, 1, 1]
    t = precision_targeted_threshold(scores, labels, 1.0)
    assert t == pytest.approx(0.9)
    flagged = [s >= t for s in scores]
    assert flagged == [False, False, True, True]


def test_fpr_targeted_threshold():
    scores = [0.1, 0.2, 0.3, 0.9]
    labels = [0, 0, 0, 1]
    t = fpr_targeted_threshold(scores, labels, 0.0)
    assert t == pytest.approx(0.9)
