"""Tests for src/data/sleepedf.py using fake in-memory recordings (no real data)."""
import mne
import numpy as np

from src.data import sleepedf

HZ = 100
EPOCH = 30


def make_fake():
    """Build ~3 hours of fake noise plus annotations covering every stage."""
    n_sec = 3 * 3600
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1e-5, size=(1, n_sec * HZ))  # volts
    info = mne.create_info(["EEG Fpz-Cz"], HZ, "eeg")
    raw = mne.io.RawArray(data, info, verbose="error")
    # (description, duration in seconds); sums to n_sec
    segs = [("Sleep stage W", 3600), ("Sleep stage 1", 300), ("Sleep stage 2", 1800),
            ("Sleep stage 3", 600), ("Sleep stage 4", 600), ("Sleep stage R", 900),
            ("Sleep stage ?", 300), ("Sleep stage W", n_sec - 3600 - 300 - 1800 - 600 - 600 - 900 - 300)]
    onsets, durs, descs, t = [], [], [], 0
    for d, dur in segs:
        onsets.append(t); durs.append(dur); descs.append(d); t += dur
    return raw, mne.Annotations(onsets, durs, descs)


def test_epochs_from_raw():
    """Shapes, stage mapping, dropped '?', wake trimming, microvolt scale."""
    raw, ann = make_fake()
    X, y = sleepedf.epochs_from_raw(raw, ann)
    assert X.shape[1:] == (1, 3000) and X.dtype == np.float32
    assert len(X) == len(y)
    assert set(np.unique(y)) == {0, 1, 2, 3, 4}
    assert (y == 3).sum() == (600 + 600) // EPOCH  # stage 3 and 4 merged
    assert (y == 1).sum() == 10 and (y == 4).sum() == 30
    # first/last epochs are wake, but at most 60 of them on each side
    first_sleep = np.argmax(y != 0)
    last_sleep = len(y) - 1 - np.argmax(y[::-1] != 0)
    assert first_sleep <= 60 and len(y) - 1 - last_sleep <= 60
    # ~1e-5 V noise -> ~10 microvolts
    assert 3 < X.std() < 30


def test_parse_name():
    """Subject and night come from the file name."""
    assert sleepedf.parse_name("/x/SC4012E0-PSG.edf") == ("SC1", 2)
    assert sleepedf.parse_name("SC4001E0-PSG.edf") == ("SC0", 1)


def test_process_writes_npz(tmp_path):
    """process() saves X, y, subject, night using a fake loader."""
    raw, ann = make_fake()
    out = tmp_path / "out.npz"
    pairs = [["SC4001E0-PSG.edf", "h"], ["SC4012E0-PSG.edf", "h"]]
    sleepedf.process(pairs, out, load_fn=lambda psg, hyp: (raw, ann))
    d = np.load(out)
    assert set(d.files) == {"X", "y", "subject", "night"}
    assert d["X"].shape[0] == len(d["y"]) == len(d["subject"]) == len(d["night"])
    assert set(d["subject"]) == {"SC0", "SC1"}
    assert set(d["night"]) == {1, 2}
