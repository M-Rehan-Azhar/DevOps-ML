# Technical Report — Social Media Virality Prediction (Full-Stack ML + CI/CD)

## 1. Objective
Deploy a trained virality classifier as a locally executable web application
(Flask + Angular), keep the code in a single Git repository on GitHub, and
automate testing, building and static deployment with GitHub Actions.

## 2. Data and target
- Source: `multi_platform_social_sentiment_evolution.csv`, 150,000 posts, 31 columns,
  no missing values or duplicate rows.
- Four engineered columns were added in the notebook (31 → 35):
  `engagement_per_view`, `engagement_velocity`, `follower_growth_rate`,
  `discussion_ratio`.
- Target: `is_viral = viral_coefficient > 2.39` (75th percentile).

## 3. Feature selection and leakage
Only information available when a post is published is used: 18 raw inputs plus
`follower_growth_rate` (followers / (account_age_days + 1)). Engagement totals
and the three engagement-derived engineered columns describe the post's
outcome, so they are excluded to avoid target leakage.

## 4. Modelling
| Model | Notes |
|---|---|
| Logistic Regression | One-hot + standard scaling, balanced class weights; baseline |
| CatBoost (selected) | Native categorical handling, depth 7, lr 0.06, l2 5, early stopping on validation AUC (200 trees) |

The threshold that maximises validation F1 (0.6003) was fixed before the single
evaluation on the held-out test set. The final model was retrained on
train + validation with the early-stopped number of trees.

Held-out test (22,500 posts): Accuracy 0.8116, Precision 0.6024, Recall 0.7218,
F1 0.6567, ROC-AUC 0.8742, PR-AUC 0.7237.

## 5. Packaging (manual Stage 4)
The notebook saved a dictionary bundle with a bare `CatBoostClassifier` plus
metadata. Because CatBoost carries its own categorical encoding there is no
separate fitted preprocessor to recover; the remaining preprocessing (engineered
feature, string casting, column order, threshold) is deterministic and is
implemented in `backend/virality_model.py` as `ViralityPipeline`, which offers
`predict` / `predict_proba` like a fitted scikit-learn pipeline (manual §4.4,
custom-model adapter). `backend/package_existing.py` wraps it into
`backend/models/model_bundle.joblib` together with the raw feature list, task,
version, class labels, threshold, test metrics and the allowed-category /
numeric-range schema measured on the training CSV.

**Verification:** on all 22,500 test rows the bundle reproduces the notebook's
probabilities exactly (max absolute difference 0.0) and therefore the stored
metrics. Sample record (first test row): probability 0.8237 → Viral in both
the notebook procedure and the API.

## 6. Backend API (Stages 6–7)
- `GET /health`, `GET /schema`, `POST /predict` (JSON only).
- Validation order: JSON object → exact key set (missing / extra reported) →
  allowed categories and training ranges → inference. Errors return HTTP 400 with
  details; unexpected failures return 500 and are logged.
- Response: `prediction`, `label`, `probabilities`, `viral_probability`,
  `threshold`, `task`, `disclaimer`.
- Tests: 9 pytest tests (manual's 4 API tests, 4 validation tests, 1 structure
  test) all pass locally and in CI.

## 7. Frontend (Stages 8–10)
- Generated with `ng new frontend --routing --style=css --skip-git --defaults`
  (Angular CLI 22.2.3).
- `feature-schema.ts` is generated from the model bundle, so the 18 form fields,
  dropdown options and numeric limits are identical to what the API accepts.
- `PredictionService` posts to `/api/predict`; `proxy.conf.json` forwards `/api`
  to Flask on 127.0.0.1:5000 during development (no CORS needed).
- Angular 22 applications are zoneless by default, so loading / error / result
  state is held in signals so the view updates when the HTTP response arrives.
- When served from any host other than localhost (GitHub Pages) the app shows a
  static-demo notice and disables prediction.

## 8. Version control and CI/CD (Stages 11–14)
- One root repository; `.gitignore` excludes `.venv`, `node_modules`, build output,
  caches, logs and the full dataset.
- The model artifacts (~0.6 MB each) and a single non-sensitive sample record are
  committed so CI can execute the model-dependent tests.
- `ci.yml`: backend (Python 3.12, pip cache, pytest incl. API tests) and frontend
  (Node 24, npm cache, `npm ci`, build, `angular-build` artifact).
- `pages.yml`: production build with `--base-href /DevOps-ML/`, upload of
  `frontend/dist/frontend/browser`, deploy to GitHub Pages.

## 9. Local results (evidence in `docs/evidence/`)
| Check | Result |
|---|---|
| Direct inference (`test_pipeline.py`) | PASS, [0.1763, 0.8237] |
| `GET /health` | 200, 18 features |
| `POST /predict` (curl and Requests) | 200, Viral, 0.8237 |
| `pytest backend/tests` | 9 passed |
| Proxy `GET localhost:4200/api/health` | 200 |
| Angular form | 18 inputs, blank field blocked, sample → Viral 82.4 % |
| Flask stopped | UI shows "API error 502" |
| Production build with Pages base-href | static notice shown, button disabled |

## 10. Limitations
- Precision 0.60: the model over-flags viral posts by design (recall-oriented F1 threshold).
- The dataset's provenance is a public Kaggle-style dataset; results may not
  transfer to real platform data.
- The public Pages site cannot run predictions; a hosted HTTPS backend would be
  needed for that.
