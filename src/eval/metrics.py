"""Scoring functions. Forecasting: MAE, MASE (lower = better). Classification: macro-F1, accuracy (higher = better)."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def mae(y_true, y_pred):
    """Mean absolute error: on average, how many bpm off the forecast is."""
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def mase(y_true, y_pred, history):
    """MAE divided by the average 1-step change in the history.

    < 1 means better than 'just repeat the previous value' measured inside the history.
    """
    scale = np.mean(np.abs(np.diff(np.asarray(history, float))))
    return mae(y_true, y_pred) / max(scale, 1e-8)


def macro_f1(y_true, y_pred):
    """F1 score per class, averaged so every class counts equally (rare classes matter too)."""
    return float(f1_score(y_true, y_pred, average="macro"))


def accuracy(y_true, y_pred):
    """Fraction of predictions that are exactly right."""
    return float(accuracy_score(y_true, y_pred))
