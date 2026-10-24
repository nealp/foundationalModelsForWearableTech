"""Raw WESAD -> processed files for Task 1 (HR forecast) and Task 2 (stress).

Reads  RAW_DIR/wesad/WESAD/<sid>/<sid>.pkl   (wrist signals only — what a watch has)
Writes PROCESSED_DIR/wesad_hr.parquet         (Task 1, Nixtla long format)
       PROCESSED_DIR/wesad_windows.npz        (Task 2, 60s windows)

Run:  python -m src.data.wesad [--quick]
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

# Sampling rates of the Empatica E4 wrist sensor and the label signal (from the WESAD readme).
WRIST_HZ = {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4}
LABEL_HZ = 700
KEEP_LABELS = {1: 0, 2: 1, 3: 0}  # baseline -> 0, stress -> 1, amusement -> 0. Everything else dropped.

# Task 2 window settings. 8 Hz (not 4) so the pulse wave (BVP) survives; 60s * 8 Hz = 480 steps.
WINDOW_HZ = 8
WINDOW_S = 60
STEP_S = 30
CHANNELS = ["BVP", "EDA", "TEMP", "ACC_X", "ACC_Y", "ACC_Z"]

START_TIME = pd.Timestamp("2017-01-01")  # fake clock start; only the 1-second spacing matters


def list_subjects(wesad_dir):
    """Return subject ids found in the WESAD folder, in numeric order (S2, S3, ..., S17)."""
    ids = [p.name for p in Path(wesad_dir).iterdir() if p.is_dir() and p.name.startswith("S")]
    return sorted(ids, key=lambda s: int(s[1:]))


def load_subject(wesad_dir, sid):
    """Load one subject's pickle file (it was saved with Python 2, hence latin1)."""
    with open(Path(wesad_dir) / sid / f"{sid}.pkl", "rb") as f:
        return pickle.load(f, encoding="latin1")


def n_seconds(data):
    """Number of whole seconds covered by every wrist signal."""
    wrist = data["signal"]["wrist"]
    return min(len(wrist[name]) // hz for name, hz in WRIST_HZ.items())


def hr_series(data):
    """Heart rate in bpm, one value per second, computed from the wrist pulse signal (BVP).

    We use BVP from the pickle (not the E4 HR.csv) because it is already time-aligned with the labels.
    """
    import neurokit2 as nk

    secs = n_seconds(data)
    bvp = data["signal"]["wrist"]["BVP"][: secs * 64, 0]
    signals, _ = nk.ppg_process(bvp, sampling_rate=64)
    rate = signals["PPG_Rate"].to_numpy().reshape(secs, 64).mean(axis=1)  # 64 samples -> 1 second
    rate[(rate < 30) | (rate > 220)] = np.nan  # impossible values = bad peaks
    return pd.Series(rate).interpolate(limit_direction="both").to_numpy()


def resample(signal, hz, target_hz, secs):
    """Bring a (n, channels) signal to `target_hz` by averaging (downsample) or repeating (upsample)."""
    signal = np.asarray(signal, dtype=float)[: secs * hz]
    if hz >= target_hz:
        factor = hz // target_hz
        return signal.reshape(secs * target_hz, factor, -1).mean(axis=1)
    return np.repeat(signal, target_hz // hz, axis=0)


def make_windows(data):
    """Cut the wrist signals into 60s windows (step 30s) at 8 Hz.

    Keeps only windows where the label is the same the whole time and is baseline/stress/amusement.
    Returns X (n_windows, 6 channels, 480), y (0 = not stressed, 1 = stressed), start_s (window start, seconds).
    """
    wrist = data["signal"]["wrist"]
    secs = n_seconds(data)
    parts = [resample(wrist[name], hz, WINDOW_HZ, secs) for name, hz in WRIST_HZ.items()]
    signal = np.concatenate([parts[1], parts[2], parts[3], parts[0]], axis=1).T  # BVP, EDA, TEMP, ACC xyz

    # label at each 8 Hz step (pick the 700 Hz label at the same moment in time)
    idx = (np.arange(secs * WINDOW_HZ) * LABEL_HZ / WINDOW_HZ).astype(int)
    labels = np.asarray(data["label"])[idx]

    X, y, starts = [], [], []
    win, step = WINDOW_S * WINDOW_HZ, STEP_S * WINDOW_HZ
    for start in range(0, signal.shape[1] - win + 1, step):
        window_labels = np.unique(labels[start : start + win])
        if len(window_labels) == 1 and window_labels[0] in KEEP_LABELS:
            X.append(signal[:, start : start + win])
            y.append(KEEP_LABELS[window_labels[0]])
            starts.append(start // WINDOW_HZ)
    return np.array(X, dtype=np.float32), np.array(y, dtype=int), np.array(starts, dtype=int)


def process(wesad_dir, subjects, hr_path, windows_path):
    """Process every subject and write the two shared files (formats in CLAUDE.md §4.4)."""
    hr_frames, Xs, ys, subs, starts = [], [], [], [], []
    for sid in subjects:
        data = load_subject(wesad_dir, sid)
        hr = hr_series(data)
        hr_frames.append(pd.DataFrame({
            "unique_id": sid,
            "ds": START_TIME + pd.to_timedelta(np.arange(len(hr)), unit="s"),
            "y": hr,
        }))
        X, y, start_s = make_windows(data)
        Xs.append(X), ys.append(y), starts.append(start_s), subs.append(np.full(len(y), sid))
        print(f"{sid}: {len(hr)} s of HR (median {np.median(hr):.0f} bpm), {len(y)} windows, {y.sum()} stressed")

    Path(hr_path).parent.mkdir(parents=True, exist_ok=True)
    pd.concat(hr_frames, ignore_index=True).to_parquet(hr_path, index=False)
    np.savez_compressed(
        windows_path,
        X=np.concatenate(Xs), y=np.concatenate(ys), subject=np.concatenate(subs),
        channels=np.array(CHANNELS), start_s=np.concatenate(starts),  # start_s: for the demo timeline
    )
    print(f"wrote {hr_path} and {windows_path}")


def main(argv=None):
    """Command line entry point. --quick processes only the first few subjects."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="only process a few subjects")
    args = parser.parse_args(argv)

    wesad_dir = config.RAW_DIR / "wesad" / "WESAD"
    subjects = list_subjects(wesad_dir)
    if args.quick:
        subjects = subjects[: config.QUICK_N_SUBJECTS]
    config.ensure_dirs()
    process(wesad_dir, subjects, config.WESAD_HR, config.WESAD_WINDOWS)


if __name__ == "__main__":
    main()
