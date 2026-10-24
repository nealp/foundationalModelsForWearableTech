"""All paths and shared settings live here. Every other file imports from this one.

Set the DATA_DIR environment variable to choose where data goes:
  - On Colab:  /content/drive/MyDrive/hackachusetts   (set by the runner notebook)
  - Locally:   leave it unset -> uses ./data in the repo (git ignores it)
"""
import os
from pathlib import Path

# The repo root is two folders up from this file (src/config.py -> repo/).
REPO_ROOT = Path(__file__).resolve().parents[1]

# Main data folder (Google Drive on Colab).
DATA_DIR = Path(os.environ.get("DATA_DIR", REPO_ROOT / "data"))

# Raw downloads go to temporary disk, never to Drive (CLAUDE.md §4.2).
RAW_DIR = Path(os.environ.get("RAW_DIR", "/content/raw" if Path("/content").exists() else DATA_DIR / "raw"))

# Folder layout from CLAUDE.md §4.3.
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"

# Files every script shares (CLAUDE.md §4.4, §4.5, §6).
WESAD_HR = PROCESSED_DIR / "wesad_hr.parquet"
WESAD_WINDOWS = PROCESSED_DIR / "wesad_windows.npz"
SLEEPEDF_EPOCHS = PROCESSED_DIR / "sleepedf_epochs.npz"
FOLDS_JSON = PROCESSED_DIR / "folds.json"
RESULTS_CSV = RESULTS_DIR / "results.csv"

# Shared settings.
SEED = 42
N_FOLDS = 5
QUICK_N_SUBJECTS = 3   # --quick mode uses this many subjects and 1 fold
HR_CONTEXT = 512       # Task 1: seconds of history the model sees
HR_HORIZON = 60        # Task 1: seconds to forecast


def ensure_dirs():
    """Create the output folders if they don't exist yet (safe to call many times)."""
    for folder in [PROCESSED_DIR, RESULTS_DIR, PREDICTIONS_DIR, CHECKPOINTS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)
