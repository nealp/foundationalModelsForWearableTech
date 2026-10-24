"""Tests for src/config.py: paths must come from the DATA_DIR env var."""
import importlib
from pathlib import Path


def load_config(monkeypatch, data_dir):
    """Reload src.config with DATA_DIR set to `data_dir`, and return the module."""
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    import src.config as config
    return importlib.reload(config)


def test_paths_follow_data_dir(monkeypatch, tmp_path):
    """Every folder lives inside DATA_DIR, matching the Drive layout in CLAUDE.md §4.3."""
    config = load_config(monkeypatch, tmp_path)
    assert config.DATA_DIR == tmp_path
    assert config.PROCESSED_DIR == tmp_path / "processed"
    assert config.RESULTS_DIR == tmp_path / "results"
    assert config.PREDICTIONS_DIR == tmp_path / "results" / "predictions"
    assert config.CHECKPOINTS_DIR == tmp_path / "checkpoints"
    assert config.RESULTS_CSV == tmp_path / "results" / "results.csv"
    assert config.FOLDS_JSON == tmp_path / "processed" / "folds.json"


def test_processed_file_names(monkeypatch, tmp_path):
    """Processed file names match the shared formats in CLAUDE.md §4.4."""
    config = load_config(monkeypatch, tmp_path)
    assert config.WESAD_HR == tmp_path / "processed" / "wesad_hr.parquet"
    assert config.WESAD_WINDOWS == tmp_path / "processed" / "wesad_windows.npz"
    assert config.SLEEPEDF_EPOCHS == tmp_path / "processed" / "sleepedf_epochs.npz"


def test_ensure_dirs_creates_folders(monkeypatch, tmp_path):
    """ensure_dirs() creates all output folders and is safe to call twice."""
    config = load_config(monkeypatch, tmp_path / "hack")
    config.ensure_dirs()
    config.ensure_dirs()
    for folder in [config.PROCESSED_DIR, config.PREDICTIONS_DIR, config.CHECKPOINTS_DIR]:
        assert folder.is_dir()


def test_default_data_dir_without_env(monkeypatch):
    """Without DATA_DIR, data goes to ./data in the repo (which .gitignore blocks)."""
    monkeypatch.delenv("DATA_DIR", raising=False)
    import src.config as config
    config = importlib.reload(config)
    assert config.DATA_DIR == Path(__file__).resolve().parents[1] / "data"


def test_shared_constants(monkeypatch, tmp_path):
    """Constants every script shares (seed, windows, folds)."""
    config = load_config(monkeypatch, tmp_path)
    assert config.SEED == 42
    assert config.HR_CONTEXT == 512
    assert config.HR_HORIZON == 60
    assert config.N_FOLDS == 5
    assert 2 <= config.QUICK_N_SUBJECTS <= 3
