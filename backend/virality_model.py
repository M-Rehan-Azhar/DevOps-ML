"""Inference adapter for the social-media virality CatBoost model.

The trained artifact (``social_media_virality_best_model.pkl``) is a dictionary
holding a fitted ``CatBoostClassifier`` plus the metadata needed to reproduce
the training-time input format. CatBoost handles categorical encoding inside
the model itself, so the only preprocessing that has to be repeated at
prediction time is:

1. derive the engineered feature ``follower_growth_rate`` from the raw inputs
   ``followers`` and ``account_age_days`` (same formula as the notebook);
2. cast the categorical columns to ``str``;
3. order the columns exactly as during training;
4. apply the decision threshold tuned on the validation set.

``ViralityPipeline`` wraps those steps behind the usual ``predict`` /
``predict_proba`` interface so the generic Flask API in ``app.py`` can use it
like a fitted scikit-learn pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ENGINEERED_FEATURE = "follower_growth_rate"


class ViralityPipeline:
    """Raw form inputs -> engineered feature -> CatBoost -> thresholded label."""

    def __init__(self, source_bundle: dict):
        self.model = source_bundle["model"]
        self.model_features = list(source_bundle["feature_columns"])
        self.categorical_features = list(source_bundle["categorical_features"])
        self.threshold = float(source_bundle["decision_threshold"])
        self.classes_ = np.array([0, 1])
        # Raw inputs = model features without the engineered column.
        self.raw_features = [f for f in self.model_features if f != ENGINEERED_FEATURE]

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [f for f in self.raw_features if f not in frame.columns]
        if missing:
            raise ValueError(f"Missing input features: {missing}")
        out = frame[self.raw_features].copy()
        for col in self.raw_features:
            if col in self.categorical_features:
                out[col] = out[col].astype(str)
            else:
                out[col] = pd.to_numeric(out[col], errors="raise").astype(float)
        out[ENGINEERED_FEATURE] = out["followers"] / (out["account_age_days"] + 1)
        return out[self.model_features]

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.transform(frame))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        viral_probability = self.predict_proba(frame)[:, 1]
        return (viral_probability >= self.threshold).astype(int)
