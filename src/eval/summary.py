"""Turn results.csv into the leaderboard table for the README / Devpost.

Run:  python -m src.eval.summary        (prints a markdown table)
"""
import pandas as pd

from src import config

# task -> (main metric, lower is better?, display name)
MAIN_METRIC = {
    "hr": ("mae", True, "Heart rate forecast"),
    "stress": ("macro_f1", False, "Stress detection"),
    "sleep": ("macro_f1", False, "Sleep staging"),
}
TIER_NAMES = {1: "1 · Simple", 2: "2 · Classic ML", 3: "3 · Neural net", 4: "4 · Foundation"}


def leaderboard(df):
    """One row per (task, model): mean and std of the main metric over folds, best first within each task."""
    rows = []
    for task, (metric, lower_better, _) in MAIN_METRIC.items():
        d = df[(df["task"] == task) & (df["metric"] == metric)]
        if d.empty:
            continue
        g = d.groupby(["model", "tier"])["value"].agg(["mean", "std", "count"]).reset_index()
        g = g.sort_values("mean", ascending=lower_better)
        rows.append(g.assign(task=task, metric=metric))
    if not rows:
        return pd.DataFrame(columns=["task", "model", "tier", "metric", "mean", "std", "count"])
    out = pd.concat(rows, ignore_index=True)
    return out[["task", "model", "tier", "metric", "mean", "std", "count"]]


def to_markdown(table):
    """Format the leaderboard as a markdown table (std is 0 when there is only one fold)."""
    lines = ["| Task | Model | Tier | Score (mean ± std over folds) | Folds |", "|---|---|---|---|---|"]
    for r in table.itertuples():
        name = MAIN_METRIC[r.task][2]
        std = 0.0 if pd.isna(r.std) else r.std
        lines.append(f"| {name} | {r.model} | {TIER_NAMES.get(r.tier, r.tier)} | {r.mean:.2f} ± {std:.2f} | {r.count} |")
    return "\n".join(lines)


def main():
    """Print the leaderboard for the current results.csv."""
    if not config.RESULTS_CSV.exists():
        raise SystemExit(f"{config.RESULTS_CSV} not found — run some jobs first")
    print(to_markdown(leaderboard(pd.read_csv(config.RESULTS_CSV))))
    print("\nHR = MAE in bpm (lower is better). Stress/Sleep = macro-F1 (higher is better).")


if __name__ == "__main__":
    main()
