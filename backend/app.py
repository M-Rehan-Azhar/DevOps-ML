from pathlib import Path
import math

import joblib
import pandas as pd
from flask import Flask, jsonify, request

BASE = Path(__file__).resolve().parent
BUNDLE = joblib.load(BASE / "models/model_bundle.joblib")
MODEL = BUNDLE["pipeline"]
FEATURES = BUNDLE["features"]
TASK = BUNDLE.get("task", "classification")
SCHEMA = BUNDLE.get("schema", {})
CLASS_LABELS = BUNDLE.get("class_labels", {0: "Not viral", 1: "Viral"})
THRESHOLD = BUNDLE.get("decision_threshold")

app = Flask(__name__)


def validate_values(data):
    """Allowed-category and numeric-range checks taken from the training data."""
    problems = {}
    for name in FEATURES:
        rule = SCHEMA.get(name)
        value = data[name]
        if rule is None:
            continue
        if rule["kind"] == "category":
            if str(value) not in rule["options"]:
                problems[name] = f"must be one of {rule['options']}"
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                problems[name] = "must be a number"
            elif not math.isfinite(value):
                problems[name] = "must be a finite number"
            elif value < rule["min"] or value > rule["max"]:
                problems[name] = f"must be between {rule['min']} and {rule['max']}"
    return problems


@app.get("/health")
def health():
    return jsonify(status="ok", features=len(FEATURES), task=TASK,
                   model_version=BUNDLE.get("model_version"))


@app.get("/schema")
def schema():
    return jsonify(features=FEATURES, schema=SCHEMA, threshold=THRESHOLD)


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Expected a JSON object"), 400
    missing = [f for f in FEATURES if f not in data]
    extra = [f for f in data if f not in FEATURES]
    if missing or extra:
        return jsonify(error="Input schema mismatch",
                       missing=missing, extra=extra), 400
    problems = validate_values(data)
    if problems:
        return jsonify(error="Invalid input values", fields=problems), 400
    frame = pd.DataFrame([[data[f] for f in FEATURES]], columns=FEATURES)
    try:
        # Model-specific inference: ViralityPipeline derives follower_growth_rate,
        # runs CatBoost and applies the validation-tuned decision threshold.
        prediction = MODEL.predict(frame)[0]
        prediction = (prediction.item() if hasattr(prediction, "item")
                      else prediction)
        result = {"prediction": prediction, "task": TASK,
                  "label": CLASS_LABELS.get(prediction, str(prediction)),
                  "disclaimer": "Research/teaching prototype only."}
        if TASK == "classification" and hasattr(MODEL, "predict_proba"):
            probabilities = MODEL.predict_proba(frame)[0].tolist()
            result["probabilities"] = probabilities
            result["viral_probability"] = probabilities[1]
            result["threshold"] = THRESHOLD
        return jsonify(result)
    except (ValueError, TypeError) as exc:
        return jsonify(error="Invalid input", detail=str(exc)), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify(error="Prediction processing failed"), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
