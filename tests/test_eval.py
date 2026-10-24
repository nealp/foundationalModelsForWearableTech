"""Tests for src/eval/metrics.py and src/eval/results.py."""
import numpy as np
import pandas as pd

from src.eval import metrics, results


def test_mae():
    """Mean absolute error of [1,2,3] vs [2,2,5] = (1+0+2)/3 = 1."""
    assert metrics.mae([1, 2, 3], [2, 2, 5]) == 1.0


def test_mase_equals_one_for_naive_on_random_walk_scale():
    """MASE = MAE / mean |step| of the history. History steps are all 2 -> scale 2."""
    history = [0, 2, 4, 6]
    assert metrics.mase([10, 10], [12, 8], history) == 1.0


def test_macro_f1_and_accuracy():
    """Perfect predictions score 1.0; macro-F1 averages over classes."""
    assert metrics.macro_f1([0, 1, 1], [0, 1, 1]) == 1.0
    assert metrics.accuracy([0, 1, 1, 0], [0, 1, 0, 0]) == 0.75
    assert np.isclose(metrics.macro_f1([0, 0, 1, 1], [0, 0, 0, 1]), (0.8 + 2 / 3) / 2)


def test_append_result_creates_then_appends(tmp_path):
    """First call writes the header; later calls add rows. Columns match CLAUDE.md §6."""
    path = tmp_path / "results" / "results.csv"
    results.append_result(path, "hr", "naive", 1, 0, "mae", 5.0, 1.2)
    results.append_result(path, "hr", "naive", 1, 1, "mae", 6.0, 1.1)
    df = pd.read_csv(path)
    assert list(df.columns) == results.COLUMNS
    assert df["value"].tolist() == [5.0, 6.0]
