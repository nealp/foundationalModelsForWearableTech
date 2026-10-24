"""Tests for src/data/wesad.py using the fake WESAD data from conftest.py."""
import numpy as np
import pandas as pd

from src.data import wesad
from tests.conftest import FAKE_SECONDS


def test_list_subjects_sorted_numerically(fake_wesad_dir):
    """S2 must come before S10 (numeric order, not text order)."""
    assert wesad.list_subjects(fake_wesad_dir) == ["S2", "S10"]


def test_hr_series_one_value_per_second(fake_wesad_dir):
    """HR is 1 Hz, has no gaps, and is close to the simulated 70 bpm."""
    data = wesad.load_subject(fake_wesad_dir, "S2")
    hr = wesad.hr_series(data)
    assert len(hr) == FAKE_SECONDS
    assert not np.isnan(hr).any()
    assert 60 < np.median(hr) < 80


def test_make_windows_shapes_and_labels(fake_wesad_dir):
    """60s windows, 30s step, 8 Hz, 6 channels; only pure baseline/stress/amusement windows kept."""
    data = wesad.load_subject(fake_wesad_dir, "S2")
    X, y, start_s = wesad.make_windows(data)
    assert X.shape == (10, len(wesad.CHANNELS), 60 * wesad.WINDOW_HZ)
    # 5 windows fit inside 0-200s (baseline) and 5 inside 200-400s (stress)
    assert y.tolist() == [0] * 5 + [1] * 5
    assert start_s[0] == 0 and start_s[5] == 210
    assert not np.isnan(X).any()


def test_process_writes_shared_formats(fake_wesad_dir, tmp_path):
    """Output files match CLAUDE.md §4.4 exactly."""
    hr_path, win_path = tmp_path / "wesad_hr.parquet", tmp_path / "wesad_windows.npz"
    wesad.process(fake_wesad_dir, ["S2", "S10"], hr_path, win_path)

    hr = pd.read_parquet(hr_path)
    assert list(hr.columns) == ["unique_id", "ds", "y"]
    assert set(hr["unique_id"]) == {"S2", "S10"}
    steps = hr[hr.unique_id == "S2"]["ds"].diff().dropna().unique()
    assert list(steps) == [pd.Timedelta(seconds=1)]

    win = np.load(win_path, allow_pickle=True)
    assert set(win.files) >= {"X", "y", "subject", "channels"}
    assert win["X"].shape[0] == len(win["y"]) == len(win["subject"]) == 20
    assert list(win["channels"]) == wesad.CHANNELS
