"""Tests for the Chronos-2 and TimesFM wrappers. Skipped if the package isn't installed.

These download real model weights (~0.5-1.3 GB) the first time, so they are slow.
Each check runs in a FRESH Python process: on macOS, torch and xgboost (loaded by other tests)
can deadlock inside one process. Real jobs are always one model per process, so this only affects tests.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CASES = [("chronos", "src.models.chronos_model"), ("timesfm3", "src.models.timesfm_model")]

CHECK = """
import numpy as np
from src import run
from tests.test_hr_pipeline import make_hr_df
model = __import__("{module}", fromlist=["forecast"])
context, target = run.make_hr_windows(make_hr_df(1), context=512, horizon=60)
pred = model.forecast(context, 60)
assert list(pred.columns) == ["unique_id", "ds", "y_hat"], pred.columns
assert len(pred) == 4 * 60, len(pred)
merged = target.merge(pred, on=["unique_id", "ds"])
assert len(merged) == 4 * 60, "timestamps must line up with the targets"
mae = np.abs(merged["y"] - merged["y_hat"]).mean()
assert mae < 10, mae
print("MAE", round(mae, 2))
"""


@pytest.mark.parametrize("package,module", CASES)
def test_forecast_shape_and_sanity(package, module):
    """Same output format as Naive/ETS, and forecasts stay near the 75 bpm series."""
    if importlib.util.find_spec(package) is None:
        pytest.skip(f"{package} not installed")
    result = subprocess.run([sys.executable, "-W", "ignore", "-c", CHECK.format(module=module)],
                            cwd=REPO, capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "MAE" in result.stdout
