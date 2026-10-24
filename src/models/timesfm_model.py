"""Tier 4 foundation model for Task 1: Google TimesFM 3.0, zero-shot (never trained on our data).

Install (in its own Colab session):  pip install "timesfm[torch]"
Model: google/timesfm-3.0-pytorch, downloaded from Hugging Face on first use.
"""
import numpy as np
import pandas as pd

from src import config

CHECKPOINT = "google/timesfm-3.0-pytorch"
_model = None  # loaded once, then re-used


def get_model():
    """Load TimesFM the first time it's needed (GPU if available, else CPU)."""
    global _model
    if _model is None:
        import torch
        from timesfm3 import ModelConfig, TimesFM3Evaluator

        torch.manual_seed(config.SEED)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = TimesFM3Evaluator(ModelConfig(checkpoint_path=CHECKPOINT, per_core_batch_size=32, device=device))
    return _model


def forecast(context_df, horizon):
    """Forecast `horizon` steps for every series in context_df (columns unique_id, ds, y).

    Returns unique_id, ds, y_hat (TimesFM's point forecast).
    """
    groups = list(context_df.groupby("unique_id", sort=False))
    contexts = [g["y"].to_numpy(dtype=np.float32) for _, g in groups]
    outputs = get_model().predict_batch(contexts, horizon=horizon, return_quantiles=False,
                                        use_symmetric_averaging=False)

    frames = []
    for (wid, g), out in zip(groups, outputs):
        # future timestamps: one second after the last context timestamp, onwards
        ds = g["ds"].iloc[-1] + pd.to_timedelta(np.arange(1, horizon + 1), unit="s")
        frames.append(pd.DataFrame({"unique_id": wid, "ds": ds, "y_hat": np.asarray(out.forecast)[:horizon]}))
    return pd.concat(frames, ignore_index=True)
