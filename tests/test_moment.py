"""Tests for the MOMENT linear probe. Shape tests always run; the real-model test is skipped
unless momentfm is installed (it lives in its own environment). Runs in a fresh process like the
other foundation-model tests (torch + xgboost can deadlock in one process on macOS)."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.models import moment_model

REPO = Path(__file__).resolve().parents[1]


def test_to_sequences_stress_one_per_channel():
    """(3 windows, 6 channels, 480) -> 18 sequences of 480."""
    seqs, parts = moment_model.to_sequences(np.zeros((3, 6, 480)), "stress")
    assert seqs.shape == (18, 480) and parts == 6


def test_to_sequences_sleep_six_chunks():
    """(2 epochs, 1, 3000) -> 12 chunks of 500, in time order."""
    X = np.arange(2 * 3000, dtype=np.float32).reshape(2, 1, 3000)
    seqs, parts = moment_model.to_sequences(X, "sleep")
    assert seqs.shape == (12, 500) and parts == 6
    assert seqs[1, 0] == 500 and seqs[6, 0] == 3000


CHECK = """
import numpy as np
from src.models import moment_model
rng = np.random.default_rng(0)
# fake stress: stressed windows are fast sine waves, calm ones are slow -> MOMENT should separate them
t = np.arange(480) / 8
def win(f): return np.stack([np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) for _ in range(6)])
X = np.stack([win(0.1) for _ in range(20)] + [win(2.0) for _ in range(20)]).astype(np.float32)
y = np.array([0] * 20 + [1] * 20)
idx = rng.permutation(40)
X, y = X[idx], y[idx]
clf = moment_model.build("stress").fit(X[:30], y[:30])
acc = (clf.predict(X[30:]) == y[30:]).mean()
assert acc >= 0.8, acc
sleep = moment_model.embed(rng.normal(size=(2, 1, 3000)).astype(np.float32), "sleep")
print("ACC", acc, "sleep emb", sleep.shape)
"""


def test_real_moment_learns_simple_pattern():
    """Real frozen MOMENT + logistic regression separates slow vs fast waves."""
    venv = REPO / ".venv-moment" / "bin" / "python"
    python = str(venv) if venv.exists() else sys.executable
    has = subprocess.run([python, "-c", "import momentfm"], capture_output=True).returncode == 0
    if not has:
        pytest.skip("momentfm not installed")
    result = subprocess.run([python, "-W", "ignore", "-c", CHECK], cwd=REPO, capture_output=True,
                            text=True, timeout=900)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "ACC" in result.stdout
