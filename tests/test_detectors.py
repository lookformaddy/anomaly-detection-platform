import random

from anomaly.detectors import CentroidDetector, IsolationForestDetector, MADDetector, ZScoreDetector
from anomaly.models import Point


def _series(seed=1, n=200):
    rng = random.Random(seed)
    return [Point(timestamp=float(i), value=100.0 + rng.gauss(0, 5)) for i in range(n)]


def test_zscore_flags_far_outlier():
    det = ZScoreDetector().fit(_series())
    assert det.score(Point(timestamp=0, value=200.0)) > 10
    assert det.score(Point(timestamp=0, value=100.0)) < 1.0


def test_mad_more_robust_than_zscore_with_contaminated_fit():
    pts = _series() + [Point(timestamp=999, value=1000.0)] * 5
    mad = MADDetector().fit(pts)
    z = ZScoreDetector().fit(pts)
    assert mad.score(Point(timestamp=0, value=200.0)) > z.score(Point(timestamp=0, value=200.0))


def test_isolation_forest_ranks_anomaly_highest():
    train = _series()
    det = IsolationForestDetector(n_trees=50, seed=123).fit(train)
    normal = det.score(Point(timestamp=0, value=100.0))
    anomaly = det.score(Point(timestamp=0, value=300.0))
    assert anomaly > normal
    assert 0.0 < anomaly <= 1.0


def test_isolation_forest_deterministic():
    train = _series()
    a = IsolationForestDetector(n_trees=20, seed=9).fit(train).score(Point(timestamp=0, value=250.0))
    b = IsolationForestDetector(n_trees=20, seed=9).fit(train).score(Point(timestamp=0, value=250.0))
    assert a == b


def test_centroid_multivariate():
    rng = random.Random(3)
    train = [Point(timestamp=float(i), value=rng.gauss(0, 1), features=(rng.gauss(10, 1),)) for i in range(100)]
    det = CentroidDetector().fit(train)
    assert det.score(Point(timestamp=0, value=0.0, features=(10.0,))) < det.score(
        Point(timestamp=0, value=20.0, features=(50.0,)))
