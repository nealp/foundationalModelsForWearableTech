"""Tier 3 forecaster for Task 1: PatchTST (Nixtla neuralforecast), trained on OUR training subjects.

Unlike the foundation models, this one needs a fit() step before forecast().
"""
import logging
import os

import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST

TIER = 3
SEED = 42
INPUT_SIZE = 512
HORIZON = 60
MAX_STEPS = int(os.environ.get("PATCHTST_MAX_STEPS", 300))  # training steps (kept small for the Colab T4 budget)

_nf = None  # the trained NeuralForecast object (set by fit)


def fit(train_df):
    """Train PatchTST on the training subjects' full HR series (columns unique_id, ds, y)."""
    global _nf
    # Re-read the env var so tests / Colab can change it after import
    max_steps = int(os.environ.get("PATCHTST_MAX_STEPS", MAX_STEPS))
    # Quiet lightning's logs
    for name in ("pytorch_lightning", "lightning", "lightning.pytorch"):
        logging.getLogger(name).setLevel(logging.ERROR)
    model = PatchTST(
        h=HORIZON,
        input_size=INPUT_SIZE,
        max_steps=max_steps,
        random_seed=SEED,
        scaler_type="standard",
        windows_batch_size=256,          # windows sampled per step (default 1024 is slow)
        val_check_steps=max_steps + 1,   # no validation set, so never validate
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
    )
    nf = NeuralForecast(models=[model], freq="s")
    nf.fit(df=train_df[["unique_id", "ds", "y"]].reset_index(drop=True), val_size=0)
    _nf = nf


def forecast(context_df, horizon):
    """Forecast the next `horizon` steps for every window in context_df (unique_id, ds, y).

    Returns a DataFrame with columns unique_id, ds, y_hat.
    """
    if _nf is None:
        raise RuntimeError("PatchTST is not trained: call neural_forecast.fit(train_df) before forecast().")
    if horizon > HORIZON:
        raise ValueError(f"PatchTST was trained with h={HORIZON}; cannot forecast horizon={horizon}.")
    pred = _nf.predict(df=context_df[["unique_id", "ds", "y"]].reset_index(drop=True))
    pred = pred.reset_index() if "unique_id" not in pred.columns else pred
    value_col = [c for c in pred.columns if c not in ("unique_id", "ds")][0]
    pred = pred.rename(columns={value_col: "y_hat"})[["unique_id", "ds", "y_hat"]]
    # Keep only the first `horizon` steps of each window
    pred = pred.groupby("unique_id", sort=False).head(horizon)
    return pred.reset_index(drop=True)
