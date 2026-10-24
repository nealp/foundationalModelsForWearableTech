"""Raw Sleep-EDF -> processed 30 s EEG epochs for Task 3 (sleep staging).

Reads  RAW_DIR/sleepedf/  (downloaded by mne's fetch_data; re-uses existing files)
Writes PROCESSED_DIR/sleepedf_epochs.npz  with X (n,1,3000), y, subject, night

Only the EEG Fpz-Cz channel (100 Hz) is used. Stages: W=0, N1=1, N2=2, N3=3
(old stages 3 and 4 merged), REM=4. "?" and "Movement time" are dropped.
Long wake periods are trimmed: we keep only from 30 min before the first sleep
epoch to 30 min after the last one (standard practice; otherwise ~70% is wake).

Run:  python -m src.data.sleepedf [--quick] [--n-subjects 20]
"""
import argparse
from pathlib import Path

import mne
import numpy as np

from src import config

HZ = 100
EPOCH_S = 30
EPOCH_LEN = HZ * EPOCH_S      # 3000 samples
TRIM_EPOCHS = 60              # 30 min of 30 s epochs
CHANNEL = "EEG Fpz-Cz"
# PhysioNet's own server was stalling (~0 KB/s); its official AWS Open Data mirror has the same files
# (~8 MB/s), and mne still checks every file's hash.
MIRROR_URL = "https://physionet-open.s3.amazonaws.com/sleep-edfx/1.0.0/sleep-cassette"
STAGES = {
    "Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
    "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4,
}


def parse_name(path):
    """Return (subject like 'SC0', night int) from e.g. SC4001E0-PSG.edf."""
    name = Path(path).name
    return f"SC{int(name[3:5])}", int(name[5])


def epochs_from_raw(raw, annotations):
    """Cut a recording into labelled 30 s epochs; returns X (n,1,3000) float32 in uV, y int."""
    signal = raw.get_data(picks=[CHANNEL])[0] * 1e6  # volts -> microvolts
    starts, labels = [], []
    for onset, dur, desc in zip(annotations.onset, annotations.duration, annotations.description):
        if desc not in STAGES:
            continue  # drops "?" and "Movement time"
        # split the long segment into consecutive 30 s epochs
        for k in range(int(dur // EPOCH_S)):
            start = int(round((onset + k * EPOCH_S) * HZ))
            if start + EPOCH_LEN > len(signal):
                break  # would run past the end of the signal
            starts.append(start)
            labels.append(STAGES[desc])
    y = np.array(labels, dtype=np.int64)
    if len(y) == 0:
        return np.zeros((0, 1, EPOCH_LEN), np.float32), y
    # trim wake: keep 30 min before the first and after the last non-wake epoch
    sleep = np.where(y != 0)[0]
    if len(sleep) == 0:
        return np.zeros((0, 1, EPOCH_LEN), np.float32), y[:0]
    lo = max(0, sleep[0] - TRIM_EPOCHS)
    hi = min(len(y), sleep[-1] + TRIM_EPOCHS + 1)
    starts, y = starts[lo:hi], y[lo:hi]
    X = np.stack([signal[s:s + EPOCH_LEN] for s in starts])[:, None, :].astype(np.float32)
    return X, y


def load_recording(psg, hyp):
    """Read one PSG file (EEG Fpz-Cz only) and its hypnogram annotations."""
    raw = mne.io.read_raw_edf(psg, include=[CHANNEL], preload=True, verbose="error")
    return raw, mne.read_annotations(hyp)


def process(file_pairs, out_path, load_fn=load_recording):
    """Process every (psg, hypnogram) pair and save one npz; load_fn lets tests fake the files."""
    Xs, ys, subjects, nights = [], [], [], []
    for psg, hyp in file_pairs:
        subject, night = parse_name(psg)
        raw, ann = load_fn(psg, hyp)
        X, y = epochs_from_raw(raw, ann)
        counts = np.bincount(y, minlength=5).tolist()
        print(f"{subject} night {night}: {len(y)} epochs, stage counts {counts}")
        Xs.append(X)
        ys.append(y)
        subjects += [subject] * len(y)
        nights += [night] * len(y)
    np.savez_compressed(out_path, X=np.concatenate(Xs), y=np.concatenate(ys),
                        subject=np.array(subjects), night=np.array(nights, dtype=np.int64))
    print(f"saved {out_path}")


def main(argv=None):
    """Command line entry point. --quick uses only a few subjects."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="only process a few subjects")
    parser.add_argument("--n-subjects", type=int, default=20, help="how many subjects to use")
    args = parser.parse_args(argv)

    n = config.QUICK_N_SUBJECTS if args.quick else args.n_subjects
    # fetch_data downloads missing files and re-uses ones already on disk
    from mne.datasets.sleep_physionet.age import fetch_data
    pairs = fetch_data(subjects=list(range(n)), recording=[1, 2],
                       path=str(config.RAW_DIR / "sleepedf"), on_missing="warn", base_url=MIRROR_URL)
    config.ensure_dirs()
    process(pairs, config.SLEEPEDF_EPOCHS)


if __name__ == "__main__":
    main()
