"""FastAPI service module. FastAPI is an *optional* dependency: importing this
module without it installed must still succeed (``app`` is None), so the core
library and its tests never depend on the web layer. Tests skip when absent."""

from __future__ import annotations

from anomaly.detectors import MADDetector, ZScoreDetector
from anomaly.models import Point
from anomaly.registry import DetectorRegistry

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except ImportError:  # pragma: no cover - exercised by the skip test
    FastAPI = None  # type: ignore
    BaseModel = object  # type: ignore

_DETECTORS = {"zscore": ZScoreDetector, "mad": MADDetector}
_registry = DetectorRegistry()
_fitted: dict[str, object] = {}


class ScoreRequest(BaseModel):  # type: ignore[misc]
    detector: str = "mad"
    value: float
    timestamp: float = 0.0


def create_app():
    if FastAPI is None:
        return None
    app = FastAPI(title="anomaly-detection-platform")

    @app.post("/fit")
    def fit(payload: dict):
        kind = payload.get("detector", "mad")
        cls = _DETECTORS.get(kind)
        if cls is None:
            return {"error": f"unknown detector {kind!r}"}
        det = cls()
        det.fit([Point(timestamp=float(t), value=float(v)) for t, v in payload.get("points", [])])
        _fitted[kind] = det
        _registry.register(kind, kind, float(payload.get("threshold", 3.5)))
        return {"detector": kind, "status": "fitted"}

    @app.post("/score")
    def score(req: ScoreRequest):
        det = _fitted.get(req.detector)
        if det is None:
            return {"error": f"detector {req.detector!r} not fitted"}
        s = det.score(Point(timestamp=req.timestamp, value=req.value))
        active = _registry.active(req.detector)
        threshold = active.threshold if active else 3.5
        return {"score": s, "is_anomaly": s >= threshold, "threshold": threshold}

    @app.get("/registry")
    def registry():
        return {"registry": _registry.to_json()}

    return app


app = create_app()
