"""Tests for hand-made features and the XGBoost baseline (all on small fake data)."""
import numpy as np

from src import config
from src.data.features import extract_features, feature_names
from src.models import xgb_baseline


def test_stress_features_shape_and_no_nan():
    """Stress features: right row count, names match columns, no NaNs."""
    X = np.random.default_rng(config.SEED).normal(size=(5, 6, 480))
    F = extract_features(X, "stress")
    assert F.shape == (5, len(feature_names("stress", 6)))
    assert not np.isnan(F).any()


def test_sleep_features_shape_and_no_nan():
    """Sleep features: right row count, names match columns, no NaNs."""
    X = np.random.default_rng(config.SEED).normal(size=(4, 1, 3000))
    F = extract_features(X, "sleep")
    assert F.shape == (4, len(feature_names("sleep", 1)))
    assert not np.isnan(F).any()


def test_alpha_sine_has_highest_relative_alpha():
    """A pure 10 Hz sine should put most of its relative power in alpha."""
    t = np.arange(3000) / 100
    X = np.sin(2 * np.pi * 10 * t)[None, None, :]
    F = extract_features(X, "sleep")[0]
    names = feature_names("sleep", 1)
    rel = {b: F[names.index(f"ch0_rel_{b}")] for b in ["delta", "theta", "alpha", "sigma", "beta"]}
    assert max(rel, key=rel.get) == "alpha"


def _fake_stress(n, rng):
    """Fake stress windows: stressed ones (y=1) have a higher EDA mean."""
    y = rng.integers(0, 2, n)
    X = rng.normal(size=(n, 6, 480))
    X[:, 1, :] += y[:, None] * 3
    return X, y


def test_stress_build_learns_separable_data():
    """XGB baseline should be >90% accurate on easy fake data."""
    rng = np.random.default_rng(config.SEED)
    Xtr, ytr = _fake_stress(80, rng)
    Xte, yte = _fake_stress(40, rng)
    m = xgb_baseline.build("stress").fit(Xtr, ytr)
    assert (m.predict(Xte) == yte).mean() > 0.9
    assert xgb_baseline.TIER == 2


def test_sleep_build_five_classes_and_missing_class():
    """Sleep model works with 5 classes, and when a train fold lacks one class."""
    rng = np.random.default_rng(config.SEED)
    X = rng.normal(size=(50, 1, 3000))
    y = np.arange(50) % 5
    pred = xgb_baseline.build("sleep").fit(X, y).predict(X)
    assert pred.shape == (50,) and set(pred) <= set(range(5))
    keep = y != 3  # drop class 3 from training
    pred = xgb_baseline.build("sleep").fit(X[keep], y[keep]).predict(X)
    assert 3 not in set(pred)
