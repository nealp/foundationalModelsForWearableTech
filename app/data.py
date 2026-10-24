"""Data helpers for the demo app (no Streamlit code in here, so it is easy to test)."""
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["task", "model", "tier", "fold", "metric", "value", "runtime_s", "timestamp"]

# Main metric per task, and whether lower is better
MAIN_METRIC = {"hr": "mae", "stress": "macro_f1", "sleep": "macro_f1"}
LOWER_IS_BETTER = {"hr": True, "stress": False, "sleep": False}

TIER_NAMES = {1: "Tier 1: Simple", 2: "Tier 2: Classic ML", 3: "Tier 3: Neural nets", 4: "Tier 4: Foundation models"}

# (task, model, tier, typical score) used to invent plausible fake numbers
_FAKE_MODELS = [
    ("hr", "naive", 1, 6.0), ("hr", "ets", 1, 5.2), ("hr", "chronos", 4, 4.1),
    ("hr", "timesfm", 4, 3.9), ("hr", "patchtst", 3, 4.6),
    ("stress", "xgboost", 2, 0.78), ("stress", "moment", 4, 0.74),
    ("sleep", "xgboost", 2, 0.62), ("sleep", "moment", 4, 0.66),
]


def make_fake_results() -> pd.DataFrame:
    """Build a made-up results table in the real results.csv format (5 folds per model)."""
    rng = np.random.default_rng(42)
    rows = []
    for task, model, tier, base in _FAKE_MODELS:
        for fold in range(5):
            noise = rng.normal(0, base * 0.05)
            rows.append([task, model, tier, fold, MAIN_METRIC[task], round(base + noise, 4),
                         round(float(rng.uniform(1, 60)), 1), "2026-01-01T00:00:00"])
    return pd.DataFrame(rows, columns=COLUMNS)


def load_results(path) -> tuple[pd.DataFrame, bool]:
    """Read the real results CSV if it exists, otherwise return fake data. Second value: is_fake."""
    path = Path(path)
    if path.exists():
        try:
            df = pd.read_csv(path)
            if len(df) > 0:
                return df, False
        except pd.errors.EmptyDataError:
            pass
    return make_fake_results(), True


def leaderboard(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """One row per model: mean and std of the task's main metric across folds, best first."""
    sub = df[(df["task"] == task) & (df["metric"] == MAIN_METRIC[task])]
    table = (sub.groupby("model")
                .agg(tier=("tier", "first"), mean=("value", "mean"), std=("value", "std"))
                .reset_index())
    table["std"] = table["std"].fillna(0.0)  # a single fold has no std
    return table.sort_values("mean", ascending=LOWER_IS_BETTER[task]).reset_index(drop=True)
