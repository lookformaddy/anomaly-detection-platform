# anomaly-detection-platform

An ML anomaly-detection serving platform built from scratch (no sklearn):
statistical and tree-based detectors, threshold calibration, streaming scoring
with windowed baselines and seasonality adjustment, drift monitoring, an alert
lifecycle, offline evaluation, and a versioned detector registry with rollback.

## The enterprise problem

Production metrics (latency, error rates, order volume, sensor readings) go
wrong in ways dashboards miss: slow drifts, joint multivariate shifts, and
level changes that look normal feature-by-feature. Teams either page on naive
fixed thresholds (alert fatigue) or hand-tune per-metric rules that rot. This
platform is the middle layer: detectors that learn normal, thresholds
calibrated to a precision/FPR target, streaming scores that re-baseline as the
world drifts, and alerts that dedup and suppress instead of paging five times
for one incident.

## Architecture overview

See [docs/architecture.md](docs/architecture.md). In short: batch detectors
(z-score, robust MAD, isolation-forest-style random-partition trees, one-class
centroid) feed a calibration step; the streaming layer seasonality-adjusts and
scores against a rolling window; a drift monitor watches the score
distribution itself; and an alert manager owns dedup, suppression, and
escalation. A FastAPI service (optional dependency) exposes `/fit`, `/score`,
and `/registry`; a versioned registry supports rollback by pointer move.

## How to run

```bash
python3 demo.py                    # end-to-end demo on synthetic data
PYTHONPATH=src python3 -m pytest -q
PYTHONPATH=src python3 -m anomaly.cli demo
PYTHONPATH=src python3 -m anomaly.cli score --detector mad --threshold 3.5
# Optional service (requires fastapi + uvicorn):
uvicorn anomaly.service:app --port 8000   # with PYTHONPATH=src
```

No network calls anywhere in core code or tests; all demo data is synthetic.

## Key design decisions and trade-offs

- **Stdlib-only core, no sklearn** — auditable, dependency-free, and the
  algorithms are small enough to implement correctly by hand.
- **Robust defaults** — MAD over z-score where fits may be contaminated;
  streaming scores use the window *before* each point so anomalies cannot
  normalise themselves.
- **Thresholds are calibrated, not hardcoded** — percentile for unlabeled
  data, precision- or FPR-targeted when labels exist, always on held-out scores.
- **Detection delay is a first-class metric** — alongside precision/recall/F1
  and from-scratch ROC-AUC — because a late flag is barely better than none.
- **Alerts separated from detection** — dedup/suppression in AlertManager, so
  tuning a detector never trades off against pager fatigue.
- **Isolation forest shines multivariate** — on univariate series MAD/z-score
  win; the forest earns its place on joint shifts.

## What it demonstrates

From-scratch ML implementation (isolation forest, ROC-AUC, PSI/KS drift),
streaming system design (windowed baselines, seasonality, drift), calibration
methodology, alert lifecycle engineering, model registry/versioning with
rollback, and a thin optional service layer — the full path from offline
evaluation to a servable, operable detector.
