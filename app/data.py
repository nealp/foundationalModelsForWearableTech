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


# ---- Stress / Sleep predictions ----

STAGE_NAMES = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
# plot position per stage (higher = higher on the axis): W, REM, N1, N2, N3 top to bottom
_HYP_POS = {0: 4, 4: 3, 1: 2, 2: 1, 3: 0}
HYP_TICKS = {4: "W", 3: "REM", 2: "N1", 1: "N2", 0: "N3"}


def load_predictions(task: str) -> dict:
    """Read <task>_<model>.parquet files (skipping *_quick) into {model: DataFrame}."""
    from src import config  # imported late so tests can reload config
    out = {}
    for f in sorted(Path(config.PREDICTIONS_DIR).glob(f"{task}_*.parquet")):
        if f.stem.endswith("_quick"):
            continue
        out[f.stem[len(task) + 1:]] = pd.read_parquet(f)
    return out


def sleep_efficiency(stages) -> float:
    """Fraction of epochs that are not wake (stage 0)."""
    stages = np.asarray(stages)
    return float((stages != 0).mean()) if len(stages) else 0.0


def hypnogram_order(stages) -> np.ndarray:
    """Map stage codes 0..4 to plot positions so the axis reads W, REM, N1, N2, N3 top to bottom."""
    return np.array([_HYP_POS[int(s)] for s in np.asarray(stages)])


def make_fake_predictions(task: str) -> dict:
    """Invent plausible predictions in the real parquet format: {model: DataFrame}."""
    rng = np.random.default_rng(0)
    out = {}
    for model, acc in [("xgboost", 0.8), ("moment", 0.75)]:
        rows = []
        for si in range(3):
            if task == "stress":
                n = 120
                y = (np.convolve(rng.random(n) < 0.08, np.ones(15), "same") > 0).astype(int)
                pred = np.where(rng.random(n) < acc, y, 1 - y)
                rows.append(pd.DataFrame({"subject": f"S{si + 2}", "start_s": np.arange(n) * 60.0,
                                          "y": y, "y_pred": pred, "fold": si}))
            else:
                n = 900
                base = np.array([0, 1, 2, 3, 2, 4]).repeat(n // 6 + 1)[:n]
                for night in range(2):
                    y = base.copy()
                    pred = np.where(rng.random(n) < acc, y, rng.integers(0, 5, n))
                    rows.append(pd.DataFrame({"subject": f"SC{si}", "night": night, "epoch": np.arange(n),
                                              "y": y, "y_pred": pred, "fold": si}))
        out[model] = pd.concat(rows, ignore_index=True)
    return out
