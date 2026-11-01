# Architecture

```
                    +-------------------+
  series points --> | StreamingScorer   | --> scores / flags
                    |  SeasonalityAdjuster (remove periodic profile)
                    |  WindowedBaseline (rolling mean/std or median/MAD)
                    +---------+---------+
                              | scores
                    +---------v---------+
                    | DriftMonitor      | PSI + KS vs reference distribution
                    +---------+---------+
                              | flags
                    +---------v---------+
                    | AlertManager      | dedup + suppression + escalation
                    +-------------------+

Offline path (training/calibration/evaluation, not on the hot path):
  fit detector (z-score / MAD / isolation forest / centroid)
    -> calibrate threshold on held-out scores (percentile / precision-targeted / FPR-targeted)
    -> evaluate: precision/recall/F1, ROC-AUC, detection delay per segment
    -> DetectorRegistry: version, activate, rollback
  FastAPI service (optional): /fit, /score, /registry -- thin wrapper, no logic.
```

## Key decisions
- **Stdlib-only core.** No sklearn: the detectors are small enough to implement
  correctly, and a from-scratch implementation is auditable and dependency-free
  in production images.
- **Scoring against the window *before* the point.** A point must not normalise
  itself; otherwise sustained anomalies silently become the new normal.
- **Robust defaults.** MAD beats z-score on contaminated fits; the streaming
  baseline defaults to the robust variant for the same reason.
- **Thresholds are calibrated, never hardcoded.** Percentile for unlabeled data;
  precision- or FPR-targeted when labels exist. The demo calibrates on held-out
  training scores, not on the evaluation set.
- **Alerts are a separate concern from detection.** Dedup/suppression live in
  AlertManager so detector tuning never has to trade off against pager fatigue.
- **Registry rollback is a pointer move.** Versions are immutable; activating
  an older version needs no redeploy, which is what makes rollback safe at 3 AM.

## Trade-offs
- Isolation forest on univariate data is weaker than MAD/z-score; it earns its
  place on multivariate series where per-feature thresholds miss joint shifts.
- PSI depends on binning, so KS is reported alongside it for drift decisions.
- The service holds fitted detectors in memory: fine for a demo and for a
  single-replica deployment, not a substitute for a persisted model store.
