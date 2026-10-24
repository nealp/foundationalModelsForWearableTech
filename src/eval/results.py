"""Append one score at a time to results.csv, so a Colab disconnect never loses more than one fold."""
from datetime import datetime
from pathlib import Path

import pandas as pd

COLUMNS = ["task", "model", "tier", "fold", "metric", "value", "runtime_s", "timestamp"]


def append_result(path, task, model, tier, fold, metric, value, runtime_s):
    """Add one row to the results CSV at `path` (creates the file with a header the first time)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([[task, model, tier, fold, metric, value, round(runtime_s, 2),
                         datetime.now().isoformat(timespec="seconds")]], columns=COLUMNS)
    row.to_csv(path, mode="a", header=not path.exists(), index=False)
