"""Shared test helpers: a tiny FAKE WESAD dataset shaped exactly like the real one.

Real WESAD: WESAD/S2/S2.pkl = {"signal": {"wrist": {...}, "chest": {...}}, "label": ..., "subject": "S2"}
Wrist rates: ACC 32 Hz (3 cols), BVP 64 Hz, EDA 4 Hz, TEMP 4 Hz. Labels at 700 Hz.
"""
import pickle

import numpy as np
import pytest

FAKE_SECONDS = 600  # 10 minutes per fake subject


def make_fake_subject(sid, seconds=FAKE_SECONDS, heart_rate=70, seed=0):
    """Build one fake subject dict. Labels: 0-200s baseline(1), 200-400s stress(2), rest 0 (dropped)."""
    import neurokit2 as nk

    rng = np.random.default_rng(seed)
    bvp = nk.ppg_simulate(duration=seconds, sampling_rate=64, heart_rate=heart_rate, random_state=seed)
    label = np.zeros(seconds * 700, dtype=int)
    label[: 200 * 700] = 1
    label[200 * 700 : 400 * 700] = 2
    wrist = {
        "ACC": rng.normal(0, 1, (seconds * 32, 3)),
        "BVP": bvp.reshape(-1, 1),
        "EDA": rng.normal(2, 0.1, (seconds * 4, 1)),
        "TEMP": rng.normal(33, 0.1, (seconds * 4, 1)),
    }
    return {"signal": {"wrist": wrist, "chest": {}}, "label": label, "subject": sid}


@pytest.fixture(scope="session")
def fake_wesad_dir(tmp_path_factory):
    """Write 2 fake subjects (S2, S10) as WESAD/<sid>/<sid>.pkl and return the WESAD folder."""
    root = tmp_path_factory.mktemp("raw") / "WESAD"
    for i, sid in enumerate(["S2", "S10"]):
        (root / sid).mkdir(parents=True)
        with open(root / sid / f"{sid}.pkl", "wb") as f:
            pickle.dump(make_fake_subject(sid, heart_rate=70 + 10 * i, seed=i), f)
    return root
