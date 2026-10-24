"""Tests for the classification path of run.py (Task 2 stress, Task 3 sleep)."""
import importlib
import json
import types

import numpy as np
import pandas as pd
import pytest


class MajorityModel:
    """Stand-in classifier: always predicts the most common training label."""

    def fit(self, X, y):
        self.label = int(np.bincount(y).argmax())
        return self

    def predict(self, X):
        return np.full(len(X), self.label)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Point DATA_DIR at a temp folder, reload config + run, and register the stand-in model."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from src import config, run
    importlib.reload(config)
    importlib.reload(run)
    config.ensure_dirs()
    fake = types.SimpleNamespace(TIER=2, build=lambda task: MajorityModel())
    monkeypatch.setitem(run.CLASSIFIERS, "majority", lambda: fake)
    return config, run


def write_stress(config, n_subjects=5, per_subject=6):
    """Fake wesad_windows.npz: 6 channels x 480 steps, half stressed per subject."""
    subjects = np.repeat([f"S{i + 2}" for i in range(n_subjects)], per_subject)
    y = np.tile([0, 0, 0, 0, 1, 1], n_subjects)
    np.savez(config.WESAD_WINDOWS, X=np.zeros((len(y), 6, 480), np.float32), y=y, subject=subjects,
             channels=np.array(["BVP", "EDA", "TEMP", "ACC_X", "ACC_Y", "ACC_Z"]),
             start_s=np.tile(np.arange(per_subject) * 30, n_subjects))
    folds = {s: i for i, s in enumerate(dict.fromkeys(subjects))}
    config.FOLDS_JSON.write_text(json.dumps({"wesad": folds}))


def test_stress_full_run_every_fold(env):
    """5 folds x 2 metrics = 10 rows; each fold tests only its own subject."""
    config, run = env
    write_stress(config)
    run.main(["--task", "stress", "--model", "majority"])
    res = pd.read_csv(config.RESULTS_CSV)
    assert len(res) == 10 and set(res["metric"]) == {"macro_f1", "accuracy"}
    assert sorted(res["fold"].unique()) == [0, 1, 2, 3, 4]
    pred = pd.read_parquet(config.PREDICTIONS_DIR / "stress_majority.parquet")
    assert {"subject", "start_s", "y", "y_pred", "fold"} <= set(pred.columns)
    assert pred.groupby("subject")["fold"].nunique().max() == 1


def test_stress_quick_trains_and_tests_on_different_people(env):
    """--quick: 1 fold, test person is never in training, written to results_quick.csv."""
    config, run = env
    write_stress(config)
    run.main(["--task", "stress", "--model", "majority", "--quick"])
    res = pd.read_csv(config.RESULTS_QUICK_CSV)
    assert len(res) == 2 and not config.RESULTS_CSV.exists()
    pred = pd.read_parquet(config.PREDICTIONS_DIR / "stress_majority_quick.parquet")
    assert pred["subject"].nunique() == 1


def test_sleep_predictions_keep_night_and_epoch_order(env):
    """Sleep predictions carry night + epoch index so the demo can draw a hypnogram."""
    config, run = env
    subjects = np.repeat(["SC0", "SC1", "SC2", "SC3", "SC4"], 10)
    np.savez(config.SLEEPEDF_EPOCHS, X=np.zeros((50, 1, 3000), np.float32), y=np.tile(np.arange(5), 10),
             subject=subjects, night=np.tile(np.repeat([1, 2], 5), 5))
    config.FOLDS_JSON.write_text(json.dumps({"sleepedf": {f"SC{i}": i for i in range(5)}}))
    run.main(["--task", "sleep", "--model", "majority"])
    pred = pd.read_parquet(config.PREDICTIONS_DIR / "sleep_majority.parquet")
    assert {"subject", "night", "epoch", "y", "y_pred"} <= set(pred.columns)
    assert pred[(pred.subject == "SC0") & (pred.night == 2)]["epoch"].tolist() == [0, 1, 2, 3, 4]
