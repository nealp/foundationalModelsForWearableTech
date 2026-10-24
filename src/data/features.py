"""Hand-made features for the classic ML baseline (XGBoost).

Turns raw windows (n, channels, time) into a flat table (n, n_features).
Everything is vectorized numpy, so ~40k sleep epochs is fine.
"""
import numpy as np
from scipy.signal import welch

# EEG frequency bands in Hz (sleep only).
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12), "sigma": (12, 15), "beta": (15, 30)}
EEG_HZ = 100
STAT_NAMES = ["mean", "std", "min", "max", "p10", "p50", "p90", "slope", "absdiff"]


def _stats(x):
    """Return the 9 summary stats of x (n, time) as a list of (n,) arrays."""
    t = np.arange(x.shape[1]) - (x.shape[1] - 1) / 2  # centered time axis
    p10, p50, p90 = np.percentile(x, [10, 50, 90], axis=1)
    slope = (x * t).sum(axis=1) / (t ** 2).sum()  # least-squares line slope
    absdiff = np.abs(np.diff(x, axis=1)).mean(axis=1)
    return [x.mean(1), x.std(1), x.min(1), x.max(1), p10, p50, p90, slope, absdiff]


def _band_features(x):
    """Return log band power then relative band power for x (n, time), as (n,) arrays."""
    freqs, psd = welch(x, fs=EEG_HZ, nperseg=200, axis=1)
    total = psd[:, (freqs >= 0.5) & (freqs < 30)].sum(axis=1) + 1e-12
    powers = [psd[:, (freqs >= lo) & (freqs < hi)].sum(axis=1) for lo, hi in BANDS.values()]
    logs = [np.log(p + 1e-12) for p in powers]
    rels = [p / total for p in powers]
    return logs + rels


def extract_features(X, task):
    """Raw windows (n, channels, time) -> float feature table (n, n_features), no NaNs."""
    X = np.asarray(X, dtype=float)
    cols = []
    for c in range(X.shape[1]):
        cols += _stats(X[:, c, :])
        if task == "sleep":
            cols += _band_features(X[:, c, :])
    if task == "stress":
        # Movement strength: vector magnitude of the 3 accelerometer channels.
        mag = np.sqrt((X[:, 3:6, :] ** 2).sum(axis=1))
        cols += [mag.mean(1), mag.std(1)]
    return np.nan_to_num(np.stack(cols, axis=1))


def feature_names(task, n_channels):
    """Column names matching extract_features, e.g. 'ch1_mean', 'ch0_rel_alpha'."""
    names = []
    for c in range(n_channels):
        names += [f"ch{c}_{s}" for s in STAT_NAMES]
        if task == "sleep":
            names += [f"ch{c}_log_{b}" for b in BANDS] + [f"ch{c}_rel_{b}" for b in BANDS]
    if task == "stress":
        names += ["acc_mag_mean", "acc_mag_std"]
    return names
