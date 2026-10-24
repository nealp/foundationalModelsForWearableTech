"""Tests for Task 1 (HR forecast): windowing, Naive/ETS models, and the run.py CLI."""
import json

import numpy as np
import pandas as pd
import pytest

from src import run
from src.models import stats_baselines


def make_hr_df(n_subjects=3, seconds=800):
    """Fake wesad_hr.parquet: a slow sine wave around 75 bpm per subject."""
    rows = []
    for i in range(n_subjects):
        t = np.arange(seconds)
        rows.append(pd.DataFrame({
            "unique_id": f"S{i + 2}",
            "ds": pd.Timestamp("2017-01-01") + pd.to_timedelta(t, unit="s"),
            "y": 75 + 5 * np.sin(t / 50) + i,
        }))
    return pd.concat(rows, ignore_index=True)


def test_make_hr_windows():
    """800s series, 512 context, 60 horizon, step 60 -> starts at 512, 572, 632, 692 -> 4 windows."""
    context, target = run.make_hr_windows(make_hr_df(1), context=512, horizon=60)
    assert context["unique_id"].nunique() == 4
    assert context.groupby("unique_id").size().unique().tolist() == [512]
    assert target.groupby("unique_id").size().unique().tolist() == [60]
    # target starts exactly one second after its context ends
    first = context["unique_id"].iloc[0]
    assert target[target.unique_id == first]["ds"].min() - context[context.unique_id == first]["ds"].max() == pd.Timedelta(seconds=1)


@pytest.mark.parametrize("model", ["naive", "ets"])
def test_stats_forecast_shape(model):
    """forecast() returns unique_id, ds, y_hat with `horizon` rows per series."""
    context, _ = run.make_hr_windows(make_hr_df(1), context=512, horizon=60)
    pred = stats_baselines.forecast(context, 60, model=model)
    assert list(pred.columns) == ["unique_id", "ds", "y_hat"]
    assert len(pred) == 4 * 60
    assert not pred["y_hat"].isna().any()


def test_naive_repeats_last_value():
    """Naive forecast = the last value of the context, repeated."""
    context, _ = run.make_hr_windows(make_hr_df(1), context=512, horizon=60)
    pred = stats_baselines.forecast(context, 60, model="naive")
    wid = context["unique_id"].iloc[0]
    last = context[context.unique_id == wid]["y"].iloc[-1]
    assert np.allclose(pred[pred.unique_id == wid]["y_hat"], last)


def test_run_cli_quick_writes_results(tmp_path, monkeypatch):
    """`python -m src.run --task hr --model naive --quick` end to end on fake data."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    import importlib
    from src import config
    importlib.reload(config)
    importlib.reload(run)
    config.ensure_dirs()
    make_hr_df(4).to_parquet(config.WESAD_HR)
    config.FOLDS_JSON.write_text(json.dumps({"wesad": {"S2": 0, "S3": 1, "S4": 2, "S5": 3}}))

    run.main(["--task", "hr", "--model", "naive", "--quick"])

    res = pd.read_csv(config.RESULTS_QUICK_CSV)
    assert set(res["metric"]) == {"mae", "mase"}
    assert (res["model"] == "naive").all() and (res["tier"] == 1).all()
    assert not config.RESULTS_CSV.exists()  # quick runs never pollute the real leaderboard
    pred = pd.read_parquet(config.PREDICTIONS_DIR / "hr_naive_quick.parquet")
    assert list(pred.columns) == ["unique_id", "window_start", "ds", "y", "y_hat"]
