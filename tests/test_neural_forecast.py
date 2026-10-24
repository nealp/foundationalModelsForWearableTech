"""Tests for the PatchTST forecaster (tiny training run so it stays fast)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src import run  # noqa: E402
pytest.importorskip("neuralforecast")  # lives in its own env (.venv-nf); skip elsewhere
from src.models import neural_forecast  # noqa: E402
from tests.test_hr_pipeline import make_hr_df  # noqa: E402


def test_forecast_before_fit_raises(monkeypatch):
    """Calling forecast without fit gives a clear error."""
    monkeypatch.setattr(neural_forecast, "_nf", None)
    context, _ = run.make_hr_windows(make_hr_df(1), context=512, horizon=60)
    with pytest.raises(RuntimeError, match="fit"):
        neural_forecast.forecast(context, 60)


def test_fit_then_forecast(monkeypatch):
    """fit + forecast returns 60 rows per window whose timestamps match the targets exactly."""
    monkeypatch.setenv("PATCHTST_MAX_STEPS", "5")
    hr = make_hr_df(3, 800)
    neural_forecast.fit(hr[hr["unique_id"] != "S4"])
    context, target = run.make_hr_windows(hr[hr["unique_id"] == "S4"], context=512, horizon=60)
    pred = neural_forecast.forecast(context, 60)
    assert list(pred.columns) == ["unique_id", "ds", "y_hat"]
    assert (pred.groupby("unique_id").size() == 60).all()
    assert pred["unique_id"].nunique() == target["unique_id"].nunique()
    merged = target.merge(pred, on=["unique_id", "ds"], how="left")
    assert len(merged) == len(target)
    assert not merged["y_hat"].isna().any()
