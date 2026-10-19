"""Stage 4 - package the trusted CatBoost artifact into model_bundle.joblib.

Run from the project root:
    python backend\\package_existing.py

Inputs
    backend/models/social_media_virality_best_model.pkl   (trained in
        notebooks/social_media_virality_final.ipynb)
    data/multi_platform_social_sentiment_evolution.csv    (optional, local
        only; used to record allowed categories and numeric ranges)

Output
    backend/models/model_bundle.joblib
"""

from pathlib import Path
import json
import pickle

import joblib
import pandas as pd

from virality_model import ViralityPipeline

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
SOURCE = ROOT / "models" / "social_media_virality_best_model.pkl"
DATASET = PROJECT / "data" / "multi_platform_social_sentiment_evolution.csv"
SCHEMA_OUT = PROJECT / "docs" / "feature_schema.json"
TASK = "classification"

with SOURCE.open("rb") as f:  # trusted artifact created by this project only
    source_bundle = pickle.load(f)

pipeline = ViralityPipeline(source_bundle)
FEATURES = pipeline.raw_features
assert hasattr(pipeline, "predict"), "Artifact must implement predict"
assert FEATURES and all(isinstance(x, str) for x in FEATURES)

schema = {}
if DATASET.exists():
    df = pd.read_csv(DATASET, usecols=FEATURES)
    for col in FEATURES:
        if col in pipeline.categorical_features:
            schema[col] = {"kind": "category",
                           "options": sorted(df[col].astype(str).unique().tolist())}
        else:
            schema[col] = {"kind": "number",
                           "min": float(df[col].min()), "max": float(df[col].max())}
    SCHEMA_OUT.parent.mkdir(parents=True, exist_ok=True)
    SCHEMA_OUT.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print("Recorded schema from", DATASET.name, "->", SCHEMA_OUT.relative_to(PROJECT))
elif SCHEMA_OUT.exists():
    schema = json.loads(SCHEMA_OUT.read_text(encoding="utf-8"))
    print("Using existing", SCHEMA_OUT.relative_to(PROJECT))

bundle = {
    "pipeline": pipeline,
    "features": FEATURES,
    "task": TASK,
    "model_version": "1.0",
    "class_labels": {0: "Not viral", 1: "Viral"},
    "decision_threshold": pipeline.threshold,
    "schema": schema,
    "test_metrics": source_bundle.get("test_metrics", {}),
    "source_artifact": SOURCE.name,
}
joblib.dump(bundle, ROOT / "models" / "model_bundle.joblib")
print("Saved model bundle:", len(FEATURES), "features")
