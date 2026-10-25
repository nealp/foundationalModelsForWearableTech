"""Tier 4 foundation model for Task 1: Amazon Chronos-2, zero-shot (never trained on our data).

Install (in its own Colab session):  pip install chronos-forecasting
Model: amazon/chronos-2 (120M parameters), downloaded from Hugging Face on first use.
"""
from src import config

MODEL_ID = "amazon/chronos-2"
_pipeline = None  # loaded once, then re-used


def get_pipeline():
    """Load Chronos-2 the first time it's needed (GPU if available, else CPU)."""
    global _pipeline
    if _pipeline is None:
        import torch
        from chronos import Chronos2Pipeline

        torch.manual_seed(config.SEED)
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        _pipeline = Chronos2Pipeline.from_pretrained(MODEL_ID, device_map=device)
    return _pipeline


def forecast(context_df, horizon):
    """Forecast `horizon` steps for every series in context_df (columns unique_id, ds, y).

    Returns unique_id, ds, y_hat, where y_hat is Chronos' median forecast.
    """
    pred = get_pipeline().predict_df(
        context_df[["unique_id", "ds", "y"]],
        prediction_length=horizon,
        quantile_levels=[0.5],
        id_column="unique_id",
        timestamp_column="ds",
        target="y",
    )
    return pred.rename(columns={"predictions": "y_hat"})[["unique_id", "ds", "y_hat"]].reset_index(drop=True)
