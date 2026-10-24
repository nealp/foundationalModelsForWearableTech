"""Tier 4 foundation model for Tasks 2 + 3: MOMENT embeddings + logistic regression ("linear probe").

MOMENT stays FROZEN — it is never trained on our data. It turns each window into a list of numbers
(an embedding); only a small logistic regression on top learns stress / sleep stages.

Install (in its own Colab session):  pip install --no-deps momentfm
  (--no-deps because momentfm pins numpy 1.25, which doesn't install on Python 3.12; it works with Colab's versions)
Model: AutonLab/MOMENT-1-base, downloaded from Hugging Face on first use.
"""
import hashlib

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config

TIER = 4
MODEL_ID = "AutonLab/MOMENT-1-base"
SEQ_LEN = 512      # MOMENT always reads 512 time steps
SLEEP_CHUNK = 500  # 30 s EEG (3000 steps) -> 6 chunks of 500, so fast brainwaves aren't blurred away
BATCH = 64

_model = None
_cache = {}  # window fingerprint -> embedding, so the 5 folds don't re-embed the same windows


def get_model():
    """Load MOMENT once in embedding mode (GPU if available, else CPU)."""
    global _model
    if _model is None:
        import torch
        from momentfm import MOMENTPipeline

        torch.manual_seed(config.SEED)
        _model = MOMENTPipeline.from_pretrained(MODEL_ID, model_kwargs={"task_name": "embedding"})
        _model.init()
        _model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return _model


def to_sequences(X, task):
    """Turn windows into single-channel sequences of <= 512 steps.

    stress: (n, 6, 480)  -> n*6 sequences of 480 (one per channel)
    sleep:  (n, 1, 3000) -> n*6 sequences of 500 (6 consecutive chunks)
    Returns (sequences (n * parts, length), parts per window).
    """
    X = np.asarray(X, dtype=np.float32)
    if task == "sleep":
        n, _, t = X.shape
        parts = t // SLEEP_CHUNK
        return X[:, 0, : parts * SLEEP_CHUNK].reshape(n * parts, SLEEP_CHUNK), parts
    n, c, t = X.shape
    return X.reshape(n * c, t), c


def embed_sequences(seqs):
    """Run frozen MOMENT on (m, length<=512) sequences -> (m, d) embeddings.

    Short sequences are padded on the left with zeros, and the input mask tells MOMENT to ignore the padding.
    """
    import torch

    model = get_model()
    device = next(model.parameters()).device
    m, length = seqs.shape
    padded = np.zeros((m, SEQ_LEN), dtype=np.float32)
    padded[:, SEQ_LEN - length:] = seqs
    mask = np.zeros((m, SEQ_LEN), dtype=np.float32)
    mask[:, SEQ_LEN - length:] = 1

    out = []
    with torch.no_grad():
        for i in range(0, m, BATCH):
            x = torch.from_numpy(padded[i:i + BATCH]).unsqueeze(1).to(device)  # (batch, 1 channel, 512)
            mk = torch.from_numpy(mask[i:i + BATCH]).to(device)
            out.append(model(x_enc=x, input_mask=mk).embeddings.float().cpu().numpy())
    return np.concatenate(out)


def embed(X, task):
    """One embedding per window: embed every channel / chunk, then join them side by side.

    Uses the cache so a window is only ever embedded once per Colab session.
    """
    keys = [hashlib.md5(np.ascontiguousarray(w).tobytes()).hexdigest() for w in np.asarray(X)]
    missing = [i for i, k in enumerate(keys) if k not in _cache]
    if missing:
        seqs, parts = to_sequences(np.asarray(X)[missing], task)
        emb = embed_sequences(seqs)
        emb = emb.reshape(len(missing), parts * emb.shape[1])
        for i, e in zip(missing, emb):
            _cache[keys[i]] = e
        print(f"MOMENT embedded {len(missing)} new windows ({len(_cache)} cached)")
    return np.stack([_cache[k] for k in keys])


class MomentProbe:
    """Frozen MOMENT embeddings + logistic regression. Same .fit/.predict interface as xgb_baseline."""

    def __init__(self, task):
        """Remember the task and set up the small classifier that will be trained."""
        self.task = task
        self.clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=config.SEED),
        )

    def fit(self, X, y):
        """Embed the training windows with frozen MOMENT, then train only the logistic regression."""
        self.clf.fit(embed(X, self.task), y)
        return self

    def predict(self, X):
        """Embed the test windows and predict labels."""
        return self.clf.predict(embed(X, self.task))


def build(task):
    """Return a fresh MOMENT linear-probe classifier for 'stress' or 'sleep'."""
    return MomentProbe(task)
