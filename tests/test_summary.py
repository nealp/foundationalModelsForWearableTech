"""Tests for src/eval/summary.py: results.csv -> leaderboard table for the README."""
import pandas as pd

from src.eval import results, summary


def test_leaderboard_markdown(tmp_path):
    """Mean ± std over folds, best model first, lower-is-better for MAE, higher for F1."""
    path = tmp_path / "results.csv"
    for fold, (a, b) in enumerate([(5.0, 3.0), (7.0, 5.0)]):
        results.append_result(path, "hr", "naive", 1, fold, "mae", a, 1)
        results.append_result(path, "hr", "chronos", 4, fold, "mae", b, 1)
    results.append_result(path, "sleep", "xgboost", 2, 0, "macro_f1", 0.70, 1)
    results.append_result(path, "sleep", "moment", 4, 0, "macro_f1", 0.60, 1)

    table = summary.leaderboard(pd.read_csv(path))
    hr = table[table.task == "hr"]
    assert hr["model"].tolist() == ["chronos", "naive"]
    assert hr["mean"].tolist() == [4.0, 6.0]
    assert table[table.task == "sleep"]["model"].tolist() == ["xgboost", "moment"]

    md = summary.to_markdown(table)
    assert "| Heart rate forecast | chronos | 4 · Foundation | 4.00 ± 1.41 | 2 |" in md
