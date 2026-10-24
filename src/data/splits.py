"""Builds processed/folds.json: which cross-validation fold each PERSON is tested in.

Split by person, never by window (CLAUDE.md §4.5). Every model reads this same file.
Format: {"wesad": {"S2": 0, "S3": 4, ...}, "sleepedf": {...}}

Run:  python -m src.data.splits
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src import config


def build_folds(subjects, n_folds=config.N_FOLDS, seed=config.SEED):
    """Assign each subject to one test fold (0..n_folds-1) with GroupKFold, groups = subject."""
    subjects = sorted(subjects)
    n_folds = min(n_folds, len(subjects))  # e.g. only 3 people after a --quick preprocess
    gkf = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = {}
    for fold, (_, test_idx) in enumerate(gkf.split(subjects, groups=subjects)):
        for i in test_idx:
            folds[subjects[i]] = fold
    return folds


def save_folds(path, dataset, folds):
    """Write one dataset's folds into folds.json, keeping any other datasets already in the file."""
    path = Path(path)
    everything = json.loads(path.read_text()) if path.exists() else {}
    everything[dataset] = folds
    path.write_text(json.dumps(everything, indent=1, sort_keys=True))


def main():
    """Build folds for every processed dataset that exists."""
    if config.WESAD_HR.exists():
        subjects = pd.read_parquet(config.WESAD_HR, columns=["unique_id"])["unique_id"].unique()
        save_folds(config.FOLDS_JSON, "wesad", build_folds(subjects))
        print(f"wesad: {len(subjects)} subjects -> {config.FOLDS_JSON}")
    if config.SLEEPEDF_EPOCHS.exists():
        subjects = np.unique(np.load(config.SLEEPEDF_EPOCHS, allow_pickle=True)["subject"])
        save_folds(config.FOLDS_JSON, "sleepedf", build_folds([str(s) for s in subjects]))
        print(f"sleepedf: {len(subjects)} subjects -> {config.FOLDS_JSON}")


if __name__ == "__main__":
    main()
