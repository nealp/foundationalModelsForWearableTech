"""Tests for the demo app (app/data.py and app/streamlit_app.py)."""
import importlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import data  # noqa: E402

COLUMNS = ["task", "model", "tier", "fold", "metric", "value", "runtime_s", "timestamp"]


def test_fake_results_format():
    """Fake results have the agreed columns, tasks and 5 folds per model."""
    df = data.make_fake_results()
    assert list(df.columns) == COLUMNS
    assert set(df["task"]) == {"hr", "stress", "sleep"}
    assert set(df["tier"]) <= {1, 2, 3, 4}
    assert (df.groupby(["task", "model"])["fold"].nunique() == 5).all()


def test_leaderboard_sorting_and_rows():
    """HR sorts by MAE ascending, others by F1 descending, one row per model."""
    df = data.make_fake_results()
    hr = data.leaderboard(df, "hr")
    assert hr["model"].is_unique and len(hr) == 5
    assert hr["mean"].is_monotonic_increasing
    for task in ["stress", "sleep"]:
        lb = data.leaderboard(df, task)
        assert lb["model"].is_unique and len(lb) == 2
        assert lb["mean"].is_monotonic_decreasing
    assert {"model", "tier", "mean", "std"} <= set(hr.columns)


def test_load_results_fallback_and_real(tmp_path):
    """Missing path -> fake data; existing CSV -> real data."""
    df, is_fake = data.load_results(tmp_path / "nope.csv")
    assert is_fake and len(df) > 0
    real = data.make_fake_results().head(3)
    path = tmp_path / "results.csv"
    real.to_csv(path, index=False)
    df, is_fake = data.load_results(path)
    assert not is_fake and len(df) == 3


def test_app_runs_without_exception(tmp_path, monkeypatch):
    """The app starts on the default page using fake data."""
    from streamlit.testing.v1 import AppTest
    from src import config

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    importlib.reload(config)
    try:
        at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=30).run()
        assert not at.exception
        # Visit every page too (HR page must cope with no prediction files)
        for page in ["HR forecast", "Stress", "Sleep"]:
            at.sidebar.radio[0].set_value(page).run()
            assert not at.exception
    finally:
        monkeypatch.undo()
        importlib.reload(config)
