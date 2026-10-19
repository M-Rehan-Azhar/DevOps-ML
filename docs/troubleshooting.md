# Troubleshooting log

| # | Step | Command / location | Error or symptom | Expected | Correction |
|---|---|---|---|---|---|
| 1 | Stage 4 | `backend/models/social_media_virality_best_model.pkl` | Artifact is a dict holding a bare CatBoost estimator, not a fitted sklearn Pipeline, and expects the engineered column `follower_growth_rate` | A loadable object with `predict()` that accepts raw inputs | Wrote `backend/virality_model.py` (`ViralityPipeline`) and `backend/package_existing.py`; verified identical probabilities on the full test split |
| 2 | Stage 8 | `npm install -g @angular/cli` | Angular CLI 22 requires Node `^22.22.3 \|\| ^24.15.0`; an older Node 22 patch release is rejected | `ng version` prints 22.x | Use Node 24 LTS (also used in CI) |
| 3 | Stage 9 | `src/app/app.ts` | Angular 22 is zoneless: plain fields changed inside an HTTP `subscribe` callback do not refresh the template | Result panel appears after the response | Hold `loading`, `error`, `result` in signals |
| 4 | Stage 9 | `src/app/app.config.ts` | Generated config no longer contains `provideZoneChangeDetection` | — | Kept the generated providers and only added `provideHttpClient()` (manual §9.3) |
| 5 | Stage 9 | Day-of-week dropdown | `day_of_week` is stored as integers but was a categorical feature in training, so dropdown values are strings `"0"`–`"6"` | Sample record pre-selects day 6 | Sample record categorical values exported as strings; API compares `str(value)` |
| 6 | Stage 7 | `pytest backend\tests` (before README existed) | `test_structure.py` failed: README.md missing | 9 passed | Created README.md, re-ran: 9 passed |
