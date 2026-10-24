"""Tests for src/data/splits.py: split by PERSON, never by window."""
import json

from src.data import splits

SUBJECTS = [f"S{i}" for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]]


def test_every_subject_in_exactly_one_fold():
    """Each subject gets one fold number in 0..4, and every fold is used."""
    folds = splits.build_folds(SUBJECTS)
    assert set(folds) == set(SUBJECTS)
    assert set(folds.values()) == {0, 1, 2, 3, 4}


def test_folds_are_balanced_and_repeatable():
    """15 subjects / 5 folds = 3 each; same seed gives the same split every time."""
    folds = splits.build_folds(SUBJECTS)
    sizes = [list(folds.values()).count(k) for k in range(5)]
    assert sizes == [3, 3, 3, 3, 3]
    assert splits.build_folds(SUBJECTS) == folds


def test_fewer_subjects_than_folds():
    """After a --quick preprocess (3 people), use 3 folds instead of crashing."""
    folds = splits.build_folds(["S2", "S3", "S4"])
    assert sorted(folds.values()) == [0, 1, 2]


def test_save_folds_keeps_other_datasets(tmp_path):
    """Writing WESAD folds must not erase Sleep-EDF folds already in folds.json."""
    path = tmp_path / "folds.json"
    splits.save_folds(path, "sleepedf", {"SC0": 0})
    splits.save_folds(path, "wesad", {"S2": 1})
    assert json.loads(path.read_text()) == {"sleepedf": {"SC0": 0}, "wesad": {"S2": 1}}
