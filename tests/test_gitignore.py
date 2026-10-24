"""Checks .gitignore blocks data files but NOT our code folders (e.g. src/data/)."""
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def is_ignored(path):
    """Return True if git would ignore `path` (path is relative to the repo root)."""
    result = subprocess.run(["git", "check-ignore", "-q", path], cwd=REPO)
    return result.returncode == 0


def test_data_is_ignored():
    """Raw/processed data and model weights must never be committed."""
    for path in ["data/x.txt", "raw/x.txt", "processed/x.txt", "checkpoints/x.txt",
                 "results/x.txt", "a.npz", "a.parquet", "a.edf", "a.pt", ".venv/x", ".venv-moment/x", ".venv-nf/x"]:
        assert is_ignored(path), path


def test_code_is_not_ignored():
    """Code folders whose names look like data folders must still be tracked."""
    for path in ["src/data/wesad.py", "src/eval/results.py", "src/config.py"]:
        assert not is_ignored(path), path
