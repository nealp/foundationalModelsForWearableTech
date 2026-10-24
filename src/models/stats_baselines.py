"""Tier 1 (simple) forecasters for Task 1, using Nixtla's statsforecast.

- naive: repeat the last value
- ets:   exponential smoothing (AutoETS picks the best trend/error type per window)
"""
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, Naive

MODELS = {"naive": Naive, "ets": AutoETS}


def forecast(context_df, horizon, model="naive"):
    """Forecast `horizon` steps for every series in context_df (columns unique_id, ds, y).

    Returns a DataFrame with columns unique_id, ds, y_hat.
    """
    sf = StatsForecast(models=[MODELS[model]()], freq="s", n_jobs=-1)
    pred = sf.forecast(h=horizon, df=context_df[["unique_id", "ds", "y"]])
    value_col = [c for c in pred.columns if c not in ("unique_id", "ds")][0]
    return pred.rename(columns={value_col: "y_hat"})[["unique_id", "ds", "y_hat"]].reset_index(drop=True)
