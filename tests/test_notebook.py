"""Checks that the Colab runner notebook is valid and has the cells from CLAUDE.md §5.1."""
import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "colab_runner.ipynb"


def all_code():
    """Return the source of every code cell joined into one string."""
    nb = json.loads(NOTEBOOK.read_text())
    assert nb["nbformat"] == 4
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def test_notebook_has_required_steps():
    """Mount Drive, clone the repo, install, set DATA_DIR, run a --quick job."""
    code = all_code()
    assert "drive.mount('/content/drive')" in code
    assert "github.com/nealp/foundationalModelsForWearableTech" in code
    assert "pip install -q -r requirements.txt" in code
    assert 'os.environ["DATA_DIR"] = "/content/drive/MyDrive/hackachusetts"' in code
    assert "--quick" in code


def test_raw_data_stays_off_drive():
    """Raw downloads must go to /content/raw (temporary disk), never to Drive."""
    code = all_code()
    assert "/content/raw" in code
    assert "MyDrive/hackachusetts/raw" not in code
