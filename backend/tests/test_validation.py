"""Extra model-specific checks: category/range validation and output shape."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app import app, SCHEMA  # noqa: E402

PAYLOAD = json.loads((ROOT / "backend/test_payload.json").read_text(encoding="utf-8"))


def test_unknown_category_rejected():
    bad = dict(PAYLOAD, platform="NotAPlatform")
    response = app.test_client().post("/predict", json=bad)
    assert response.status_code == 400
    assert "platform" in response.get_json()["fields"]


def test_out_of_range_number_rejected():
    bad = dict(PAYLOAD, hour_of_day=SCHEMA["hour_of_day"]["max"] + 10)
    response = app.test_client().post("/predict", json=bad)
    assert response.status_code == 400


def test_extra_field_rejected():
    bad = dict(PAYLOAD, is_viral=1)  # the target must never be sent
    response = app.test_client().post("/predict", json=bad)
    assert response.status_code == 400
    assert response.get_json()["extra"] == ["is_viral"]


def test_prediction_output_shape():
    body = app.test_client().post("/predict", json=PAYLOAD).get_json()
    assert body["prediction"] in (0, 1)
    assert len(body["probabilities"]) == 2
    assert abs(sum(body["probabilities"]) - 1) < 1e-6
    assert body["prediction"] == int(body["viral_probability"] >= body["threshold"])
