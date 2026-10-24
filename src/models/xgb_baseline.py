"""Tier-2 baseline: hand-made features + XGBoost.

Shared classifier interface: build(task) -> object with .fit(X, y) and .predict(X),
where X is the RAW window array (features are computed inside).
"""
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src import config
from src.data.features import extract_features

TIER = 2


class XGBWindowClassifier:
    """Feature extraction + class-weighted XGBoost, robust to missing classes."""

    def __init__(self, task):
        """Remember the task ('stress' or 'sleep') and create the XGBoost model."""
        self.task = task
        self.encoder = LabelEncoder()
        self.model = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, random_state=config.SEED, n_jobs=-1, tree_method="hist",
        )

    def fit(self, X, y):
        """Train on raw windows X and labels y; returns self."""
        y_enc = self.encoder.fit_transform(y)  # relabel to 0..k-1 so missing classes are OK
        weights = compute_sample_weight("balanced", y_enc)
        self.model.fit(extract_features(X, self.task), y_enc, sample_weight=weights)
        return self

    def predict(self, X):
        """Predict integer labels (original label values) for raw windows X."""
        pred = self.model.predict(extract_features(X, self.task))
        return self.encoder.inverse_transform(np.asarray(pred, dtype=int)).astype(int)


def build(task):
    """Return an untrained model for the given task."""
    return XGBWindowClassifier(task)
