# Social Media Virality Prediction — Full-Stack ML with CI/CD

A trained CatBoost classifier served by a local **Flask REST API**, used from an
**Angular 22** feature form, with **GitHub Actions** for automated tests and builds
and a static **GitHub Pages** deployment of the frontend.

| | |
|---|---|
| GitHub repository | https://github.com/M-Rehan-Azhar/DevOps-ML |
| GitHub Pages (static UI only) | https://m-rehan-azhar.github.io/DevOps-ML/ |

> Research / teaching prototype. The score is a statistical estimate from a
> synthetic-style public dataset and is not a guarantee of real-world reach.

---

## 1. Problem and task

**Binary classification** — predict whether a social-media post will be *viral*
from information known at posting time (platform, timing, account, content and
sentiment), *before* any engagement is observed.

- **Target `is_viral`**: 1 when the post's `viral_coefficient` is above the dataset's
  75th percentile (threshold **2.39**), otherwise 0 (≈ 25 % positive class).
- **Output**: `prediction` (0 = Not viral, 1 = Viral), `label`, class
  `probabilities`, `viral_probability` and the decision `threshold`.

## 2. Model provenance

| Item | Value |
|---|---|
| Training notebook | `notebooks/social_media_virality_final.ipynb` |
| Dataset | `multi_platform_social_sentiment_evolution.csv` — 150,000 rows × 31 columns (kept local, not committed) |
| Split | 70 % train / 15 % validation / 15 % test, stratified, `random_state=42` |
| Candidates | Logistic Regression (baseline) vs **CatBoost** (selected) |
| Final model | `CatBoostClassifier` (depth 7, lr 0.06, 200 trees from early stopping, balanced class weights), retrained on train + validation |
| Decision threshold | **0.6003**, tuned for F1 on the validation set |
| Held-out test metrics | Accuracy 0.8116 · Precision 0.6024 · Recall 0.7218 · F1 0.6567 · ROC-AUC 0.8742 · PR-AUC 0.7237 |
| Trusted source artifact | `backend/models/social_media_virality_best_model.pkl` (created by the notebook) |
| Deployable bundle | `backend/models/model_bundle.joblib` (created by `backend/package_existing.py`) |
| Key libraries | Python 3.12, catboost 1.2.10, pandas, numpy, scikit-learn, Flask 3.1 |

The packaged bundle was checked against the notebook procedure on the full
22,500-row test split: **identical probabilities (max difference 0.0) and identical
labels**, reproducing the stored test metrics exactly.

### Preprocessing kept with the model

CatBoost encodes categorical features internally, so no external encoder or
scaler is needed. `backend/virality_model.py` (`ViralityPipeline`) repeats the
training-time steps:

1. derive `follower_growth_rate = followers / (account_age_days + 1)`;
2. cast the 7 categorical columns to `str`;
3. order columns exactly as in training;
4. apply the 0.6003 decision threshold.

**Leakage control:** engagement outcomes (`likes`, `shares`, `comments`, `views`,
`total_engagement`, `engagement_rate_per_1k_followers`, `viral_coefficient`,
`cross_platform_spread` and the engagement-derived features) are never inputs.

## 3. Input schema (18 raw features)

The full schema with allowed categories and numeric ranges recorded from the
training data is in [`docs/feature_schema.json`](docs/feature_schema.json); the
API rejects unknown categories and out-of-range numbers.

| # | Feature | Type | Allowed values / range |
|---|---|---|---|
| 1 | platform | category | Facebook, Instagram, Reddit, TikTok, Twitter, YouTube |
| 2 | hour_of_day | number | 1 – 23 |
| 3 | day_of_week | category | 0 – 6 |
| 4 | is_weekend | number | 0 / 1 |
| 5 | followers | number | 0 – 148,605,537 |
| 6 | account_age_days | number | 30 – 3,649 |
| 7 | verified | number | 0 / 1 |
| 8 | topic | category | AI/ML, Business, Climate, Education, Entertainment, Fashion, Finance, Food, Gaming, Health, Politics, Science, Sports, Technology, Travel |
| 9 | language | category | Arabic, Chinese, English, French, German, Hindi, Japanese, Korean, Portuguese, Spanish |
| 10 | content_length | number | 10 – 499 |
| 11 | media_type | category | Image, Link, Poll, Text, Video |
| 12 | num_hashtags | number | 0 – 10 |
| 13 | sentiment_category | category | Negative, Neutral, Positive |
| 14 | sentiment_positive | number | 0.01 – 0.95 |
| 15 | sentiment_negative | number | -0.149 – 0.95 |
| 16 | sentiment_neutral | number | -0.099 – 0.899 |
| 17 | toxicity_score | number | 0 – 100 |
| 18 | location | category | Africa, Asia, Europe, North America, Oceania, South America, Unknown |

Engineered (computed server-side, not entered): `follower_growth_rate`.

**Sample record** (`data/sample_inputs.csv` = first row of the notebook's test
split, target removed): YouTube · Sports · English · Image · 7,774 followers →
**Viral, probability 0.8237** — the same value the notebook produces for that row.

## 4. Architecture and deployment boundary

```
Browser / Angular form (http://localhost:4200)
   | POST /api/predict (JSON)
   v
Angular dev proxy -> Flask API (http://127.0.0.1:5000)
   | schema + range validation -> ViralityPipeline -> CatBoost
   v
JSON prediction -> Angular result panel

Git repository -> GitHub Actions (pytest + Angular build)
               -> GitHub Pages (STATIC ANGULAR ONLY)
```

GitHub Pages hosts only the compiled Angular files. It does **not** run Python,
Flask or the model; on Pages the form renders but the prediction button is
disabled with a notice. Live public predictions would require a separately
hosted, secured HTTPS backend.

## 5. Requirements

- Windows 10/11, VS Code (Python + Pylance extensions)
- Python **3.12**
- Node.js **24 LTS** (Angular CLI 22 needs Node `^22.22.3 || ^24.15.0`) with npm
- Git for Windows
- Versions used for verification: [`docs/environment.txt`](docs/environment.txt)

## 6. Installation (PowerShell, from the project root)

```powershell
cd E:\DevOps\DevOps-ML
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt

cd frontend
npm ci
cd ..
```

Optional — rebuild the bundle, sample payload and Angular schema from the
original artifact (place the full CSV in `data/` first):

```powershell
python backend\package_existing.py
python backend\make_payload.py
python backend\export_frontend_schema.py
```

## 7. Running locally (three terminals)

| Terminal | Folder | Command | Keep open? |
|---|---|---|---|
| 1 — Flask | project root | `.\.venv\Scripts\Activate.ps1` then `python backend\app.py` | Yes |
| 2 — Angular | `frontend` | `ng serve --open --proxy-config proxy.conf.json` | Yes |
| 3 — Commands | project root | git, pytest, curl, builds | Use freely |

Open http://localhost:4200, click **Fill sample record**, then **Run local prediction**.

### API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | `{"status":"ok","features":18,...}` |
| GET | `/schema` | feature list, allowed categories / ranges, threshold |
| POST | `/predict` | JSON object with the 18 features → prediction |

## 8. Tests (Terminal 3)

```powershell
python backend\test_pipeline.py                     # direct inference, expect PASS: local inference
curl.exe --noproxy "*" -i http://127.0.0.1:5000/health
curl.exe --noproxy "*" -i -X POST "http://127.0.0.1:5000/predict" -H "Content-Type: application/json" --data-binary "@backend/test_payload.json"
python backend\test_api.py                          # requires Flask running, expect PASS: API request
python -m pytest -q backend\tests                   # 9 passed
curl.exe --noproxy "*" -i http://localhost:4200/api/health   # via the Angular proxy
cd frontend; npm run build; npx ng test --watch=false; cd ..
```

`backend/tests/test_api.py` holds the manual's four API tests; `test_validation.py`
adds category/range/target-leak checks and `test_structure.py` the portable smoke
test. Saved outputs and screenshots are in [`docs/evidence/`](docs/evidence/).

## 9. CI/CD

- **`.github/workflows/ci.yml` — Full-stack CI** (push / PR to `main`, manual):
  backend job installs `requirements.txt` and runs the structure test **and** the
  model-dependent API tests (the approved model bundle and sample payload are
  tracked, so the runner can load the model); frontend job runs `npm ci`,
  `npm run build` and uploads the `angular-build` artifact.
- **`.github/workflows/pages.yml` — Deploy Angular to GitHub Pages**: production
  build with `--base-href /DevOps-ML/`, uploads `frontend/dist/frontend/browser`
  and deploys it. Requires *Settings → Pages → Source: GitHub Actions*.

## 10. Repository layout

```
DevOps-ML/
├── .github/workflows/ci.yml, pages.yml
├── backend/
│   ├── app.py                    Flask API
│   ├── virality_model.py         inference adapter (ViralityPipeline)
│   ├── package_existing.py       .pkl -> model_bundle.joblib
│   ├── make_payload.py           sample CSV -> test_payload.json
│   ├── export_frontend_schema.py bundle -> Angular feature-schema.ts
│   ├── test_pipeline.py, test_api.py, test_payload.json
│   ├── requirements.txt
│   ├── models/                   social_media_virality_best_model.pkl, model_bundle.joblib
│   └── tests/                    test_api.py, test_validation.py, test_structure.py
├── data/sample_inputs.csv        one non-sensitive record (no target)
├── docs/                         environment, schema, report, troubleshooting, evidence
├── frontend/                     Angular 22 app (proxy.conf.json, src/app/...)
└── notebooks/social_media_virality_final.ipynb
```

## 11. Limitations and ethics

- Predictions estimate the *probability of top-quartile virality* for this dataset;
  they are not a promise of engagement and should not be used to target people.
- Precision is ≈ 0.60 at the chosen threshold: roughly 4 in 10 posts flagged viral are not.
- No personal data is published: the sample record contains no post or user IDs,
  and the full dataset is excluded by `.gitignore`.
- The Flask server is a development server bound to `127.0.0.1`; it is not hardened
  for public deployment.

Further detail: [`docs/technical_report.md`](docs/technical_report.md) ·
[`docs/troubleshooting.md`](docs/troubleshooting.md)
