"""Run one model on one task and save its scores.

Examples:
  python -m src.run --task hr --model naive --quick     # fast test: 3 subjects, 1 fold
  python -m src.run --task hr --model ets               # full run: all folds

Scores go to results/results.csv (or results_quick.csv with --quick), one row per fold and metric.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from src import config
from src.eval import metrics, results

# name -> (tier, function that loads the model's forecast function).
# Loading lazily means you only need the packages for the model you run.
FORECASTERS = {
    "naive": (1, lambda: _stats("naive")),
    "ets": (1, lambda: _stats("ets")),
}


def _stats(name):
    """Return a forecast(context_df, horizon) function for a statsforecast model."""
    from src.models import stats_baselines
    return lambda context_df, horizon: stats_baselines.forecast(context_df, horizon, model=name)


def make_hr_windows(hr_df, context=config.HR_CONTEXT, horizon=config.HR_HORIZON, step=config.HR_HORIZON):
    """Cut each subject's HR series into (context, target) windows, one every `step` seconds.

    Window ids look like "S2|512" (subject | second where the forecast starts).
    Returns two long-format DataFrames: context (unique_id, ds, y) and target (unique_id, ds, y).
    """
    contexts, targets = [], []
    for sid, series in hr_df.groupby("unique_id", sort=False):
        ds, y = series["ds"].to_numpy(), series["y"].to_numpy()
        for t in range(context, len(y) - horizon + 1, step):
            wid = f"{sid}|{t}"
            contexts.append(pd.DataFrame({"unique_id": wid, "ds": ds[t - context:t], "y": y[t - context:t]}))
            targets.append(pd.DataFrame({"unique_id": wid, "ds": ds[t:t + horizon], "y": y[t:t + horizon]}))
    return pd.concat(contexts, ignore_index=True), pd.concat(targets, ignore_index=True)


def score_hr(context, target, pred):
    """Average MAE and MASE over all windows. Returns (scores dict, merged true+predicted DataFrame)."""
    merged = target.merge(pred, on=["unique_id", "ds"], how="left")
    assert not merged["y_hat"].isna().any(), "model returned fewer predictions than expected"
    history = dict(tuple(context.groupby("unique_id")["y"]))
    maes, mases = [], []
    for wid, w in merged.groupby("unique_id"):
        maes.append(metrics.mae(w["y"], w["y_hat"]))
        mases.append(metrics.mase(w["y"], w["y_hat"], history[wid]))
    return {"mae": float(np.mean(maes)), "mase": float(np.mean(mases))}, merged


def load_folds(dataset):
    """Read {subject: fold} for one dataset from folds.json."""
    if not config.FOLDS_JSON.exists():
        raise SystemExit(f"{config.FOLDS_JSON} not found. Run: python -m src.data.splits")
    return json.loads(config.FOLDS_JSON.read_text())[dataset]


def run_hr(model, quick=False, max_windows=None):
    """Task 1: zero-shot / per-window forecasting, scored on each fold's test subjects."""
    tier, load = FORECASTERS[model]
    forecast = load()
    hr = pd.read_parquet(config.WESAD_HR)
    folds = load_folds("wesad")

    if quick:  # first few subjects, all treated as one fold
        subjects = sorted(folds, key=lambda s: int(s[1:]))[: config.QUICK_N_SUBJECTS]
        fold_subjects = {0: subjects}
    else:
        fold_subjects = {k: [s for s, f in folds.items() if f == k] for k in sorted(set(folds.values()))}

    out_csv = config.RESULTS_QUICK_CSV if quick else config.RESULTS_CSV
    pred_path = config.PREDICTIONS_DIR / f"hr_{model}{'_quick' if quick else ''}.parquet"
    all_preds = []
    for fold, subjects in fold_subjects.items():
        start = time.time()
        context, target = make_hr_windows(hr[hr["unique_id"].isin(subjects)])
        if max_windows:  # evenly spaced sample of windows, to keep slow models fast
            ids = context["unique_id"].unique()
            keep = ids[np.linspace(0, len(ids) - 1, min(max_windows, len(ids))).astype(int)]
            context, target = context[context.unique_id.isin(keep)], target[target.unique_id.isin(keep)]
        pred = forecast(context, config.HR_HORIZON)
        scores, merged = score_hr(context, target, pred)
        runtime = time.time() - start

        # save after EVERY fold (CLAUDE.md §5.2)
        for metric, value in scores.items():
            results.append_result(out_csv, "hr", model, tier, fold, metric, value, runtime)
        ids = merged["unique_id"].str.split("|", expand=True)
        merged = merged.assign(unique_id=ids[0], window_start=ids[1].astype(int))
        all_preds.append(merged[["unique_id", "window_start", "ds", "y", "y_hat"]])
        pd.concat(all_preds, ignore_index=True).to_parquet(pred_path, index=False)
        n = context["unique_id"].nunique()
        print(f"fold {fold}: {len(subjects)} subjects, {n} windows, "
              f"MAE {scores['mae']:.2f} bpm, MASE {scores['mase']:.2f} ({runtime:.0f}s)")
    print(f"scores -> {out_csv}\npredictions -> {pred_path}")


def main(argv=None):
    """Parse command line arguments and run the chosen task + model."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=["hr", "stress", "sleep"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--quick", action="store_true", help="3 subjects, 1 fold — always try this first")
    parser.add_argument("--max-windows", type=int, default=None, help="hr only: cap windows per fold")
    args = parser.parse_args(argv)

    np.random.seed(config.SEED)
    config.ensure_dirs()
    if args.task == "hr":
        if args.model not in FORECASTERS:
            raise SystemExit(f"unknown hr model {args.model!r}; choose from {sorted(FORECASTERS)}")
        run_hr(args.model, quick=args.quick, max_windows=args.max_windows)
    else:
        raise SystemExit(f"task {args.task!r} arrives in Stage 3")


if __name__ == "__main__":
    main()
