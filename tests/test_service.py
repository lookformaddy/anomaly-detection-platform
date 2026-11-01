import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from anomaly.service import create_app


def test_score_endpoint_requires_fit_first():
    client = TestClient(create_app())
    resp = client.post("/score", json={"detector": "mad", "value": 100.0})
    assert resp.status_code == 200
    assert "not fitted" in resp.json()["error"]


def test_fit_then_score():
    client = TestClient(create_app())
    points = [{"timestamp": float(i), "value": 100.0} for i in range(50)]
    assert client.post("/fit", json={"detector": "mad", "points": points}).json()["status"] == "fitted"
    resp = client.post("/score", json={"detector": "mad", "value": 500.0})
    assert resp.json()["is_anomaly"] is True
