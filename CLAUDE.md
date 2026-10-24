# Foundation Models for Wearables — Project Plan (Hackachusetts)

> **For AI coding agents (Claude Code / Kiro):** This file is the source of truth. Read it fully before every task.
> Follow the **Agent Rules** section. Work on ONE small task at a time. If something here is unclear, ask the human instead of guessing.
>
> **Kiro users:** copy this file to `.kiro/steering/project.md` so Kiro follows the same rules.

---

## 0. Today's Schedule (READ FIRST)

**Hard deadline: 8:00 PM. Our target: done by 7:00 PM.** We started at ~3:00 PM, so we have **~4 hours**.
The full plan below (§1–§11) is too big for 4 hours. We build in **stages**. Each stage ends with something
that works and could be submitted. **Never start a stage until the one before it works.**

**Scope cuts for today (do NOT build unless every stage is done early):**
- 🟡 **PatchTST on Task 1 = first thing to add back** if Stage 3 is done by ~5:00 PM (`neuralforecast`, same in/out as ETS, ~30–45 min). Keeps a neural net on the leaderboard.
- ❌ DeepAR, 1D-CNN/LSTM (rest of Tier 3 / M4) → only if Stage 4 finishes before 6:00 PM
- ❌ ARIMA (slow on many windows) → ETS + Naive are enough for Tier 1
- ❌ MMASH cortisol, fine-tuning, W&B (all Stretch)
- ⚠️ Sleep-EDF: use **10 people**, not 20, if the download/preprocessing is slow
- ⚠️ Foundation models on Task 1: evaluate on a **sample of windows** (e.g. 50 per subject) if runs take > 15 min

### Stage 1 — Setup (3:00 → 3:30) · everyone
- P2: repo skeleton (§6 folders), `.gitignore`, `requirements.txt`, `src/config.py`, push to GitHub
- P3: `notebooks/colab_runner.ipynb` that mounts Drive + clones repo
- P1: **start the WESAD and Sleep-EDF downloads in Colab NOW** (they take the longest)
- ✅ Done when: Colab can clone the repo, mount Drive, and `import src.config` works

### Stage 2 — Data + first result (3:30 → 4:30)
- P1: `wesad.py` → `wesad_hr.parquet` + `wesad_windows.npz`; `splits.py` → `folds.json`
- P2: `run.py` CLI + `metrics.py` + `results.py` + **Naive** and **ETS** on Task 1 (test on fake data until P1 is done)
- P3: Streamlit app skeleton (leaderboard page) using a **fake** `results.csv`
- ✅ Done when: first real rows are in `results.csv` (Task 1, Naive + ETS)

### Stage 3 — Foundation models on HR + stress baseline (4:30 → 5:30)
- P2: **Chronos** on Task 1 (easiest FM to install), then **TimesFM**
- P1: `features.py` (simple stats per channel) → P2 runs **XGBoost** on Task 2
- P1: `sleepedf.py` → `sleepedf_epochs.npz` (in parallel, on a different Colab account)
- P3: HR forecast page (real HR + forecasts) as soon as `predictions/` has data
- ✅ Done when: Task 1 has Naive, ETS, Chronos, TimesFM; Task 2 has XGBoost
- 🟢 **This is our minimum submittable project.** If we are behind at 5:30, skip to Stage 5.

### Stage 4 — MOMENT + sleep (5:30 → 6:20)
- P2: **MOMENT** embeddings + logistic regression on Task 2, then Task 3
- P2: **XGBoost** on Task 3
- P3: stress timeline page + sleep hypnogram page + sleep efficiency score
- ✅ Done when: all 3 tasks have a baseline and a foundation model in `results.csv`

### Stage 5 — Ship (6:20 → 7:00) · CODE FREEZE at 6:20
- P3: record the demo video (§10), README with results table + GIF
- P1 + P2: fill the real numbers into README / §11, help with Devpost write-up
- ✅ Done when: Devpost submitted. **7:00–8:00 is buffer only** — for fixes, not new features.

### Is 7 PM realistic?
Yes, **with the cuts above**. The biggest risks are (1) dataset downloads, (2) installing
TimesFM/MOMENT on Colab, (3) Colab disconnecting. That's why downloads start in Stage 1, Chronos goes
before TimesFM, and every run uses `--quick` first and saves after every fold.

---

## 1. The Idea (one sentence)

Can new **time-series foundation models** (TimesFM, Chronos, MOMENT) — used mostly *out of the box* — read wearable sensor data better than the **classic methods** and **neural networks trained directly on that data**?

**Pitch:** "Your watch says you're stressed. How good is that guess — and can a model that has never seen wearable data do better?"

---

## 2. The 3 Tasks

| # | Task | What we predict | Type | Dataset | Main metric |
|---|------|-----------------|------|---------|-------------|
| 1 | **Heart rate forecast** | Next 60 seconds of heart rate | Forecasting | WESAD (wrist) | MAE (lower = better) |
| 2 | **Stress detection** | Stressed vs not stressed (per 60s window) | Classification (2 classes) | WESAD (wrist) | Macro-F1 (higher = better) |
| 3 | **Sleep staging** | Wake / N1 / N2 / N3 / REM (per 30s) | Classification (5 classes) | Sleep-EDF (EEG) | Macro-F1 (higher = better) |

From sleep staging we also compute a simple **sleep quality score** per night (sleep efficiency = time asleep / time in bed) for the demo.

**Stretch:** cortisol prediction using the MMASH dataset (small, ~22 people). Only if everything else is done.

---

## 3. The Models (4 tiers)

| Tier | Models | Trained on our data? | Used for |
|------|--------|----------------------|----------|
| 1. Simple | Naive (repeat last value), ARIMA, ETS | Fitted per series | Task 1 |
| 2. Classic ML | XGBoost on hand-made features | Yes | Tasks 2, 3 |
| 3. Neural nets | PatchTST, DeepAR (Task 1); small 1D-CNN or LSTM (Tasks 2, 3) | Yes | All |
| 4. Foundation models | TimesFM, Chronos (Task 1); MOMENT embeddings + simple classifier (Tasks 2, 3) | **No** (frozen) | All |

> **Today (see §0):** ARIMA, DeepAR and CNN/LSTM are cut. PatchTST (Task 1) is added back first if we're ahead.

Notes:
- Tier 1 + PatchTST + DeepAR come from the TimesFM paper's own baselines.
- For classification, foundation models stay **frozen**: we extract embeddings and train only a small logistic regression on top ("linear probe"). Say this clearly in the video — it's still fair.
- Use the **newest** TimesFM release on Hugging Face and the **small/base** sizes of every foundation model (Colab GPU limits).

**Libraries (check each library's README for the current API before coding — these change often):**
- `statsforecast` (ARIMA, ETS, Naive), `neuralforecast` (PatchTST, DeepAR) — both from Nixtla
- `timesfm`, `chronos-forecasting`, `momentfm`
- `xgboost`, `scikit-learn`, `torch`
- `mne` (Sleep-EDF loading), `neurokit2` (HR/HRV from wrist signal if needed)
- `pandas`, `pyarrow`, `numpy`, `matplotlib`, `streamlit`

---

## 4. Data Management

### 4.1 Datasets

**WESAD** (stress + heart rate)
- 15 people, Empatica E4 wrist sensor + labels.
- Use the **wrist** signals only (that's what a consumer watch has): BVP, EDA, TEMP, ACC.
- Heart rate: use the E4's 1 Hz HR file if present in each subject's E4 folder; otherwise derive HR from BVP with `neurokit2`.
- Labels: 1 = baseline, 2 = stress, 3 = amusement. **Drop everything else.** Stress task = label 2 vs labels {1, 3}.
- Download link: get it from the official WESAD page (human provides the link; agents don't guess URLs).

**Sleep-EDF** (sleep)
- Use the **sleep-cassette** subset, **first 20 people only** (10 if short on time — see §0).
- Load with `mne.datasets.sleep_physionet.age.fetch_data(...)`.
- Use the **Fpz-Cz EEG** channel. 30-second epochs. Merge stages 3 + 4 into N3. Drop movement/unknown epochs.

### 4.2 The Golden Rule

**Raw data never goes to Google Drive or GitHub.**
1. Download raw data into Colab's temporary disk (`/content/raw/`).
2. Run the preprocessing script once.
3. Save only the small processed files to Drive.
4. Every other script loads ONLY the processed files.

### 4.3 Folder Layout on Google Drive

```
MyDrive/hackachusetts/
├── processed/
│   ├── wesad_hr.parquet          # Task 1
│   ├── wesad_windows.npz         # Task 2
│   └── sleepedf_epochs.npz       # Task 3
├── results/
│   ├── results.csv               # ALL scores, one row per (task, model, fold)
│   └── predictions/              # sample predictions for the demo
└── checkpoints/                  # trained neural nets (small)
```

### 4.4 Shared Data Formats (DO NOT CHANGE without telling the team)

**Task 1 — `wesad_hr.parquet`** (the Nixtla "long" format)
| column | meaning |
|--------|---------|
| `unique_id` | subject id, e.g. `S2` |
| `ds` | timestamp (1-second steps) |
| `y` | heart rate (bpm) |

- Context = 512 seconds of history, horizon = 60 seconds ahead.
- Make forecast windows every 60 seconds along each subject's series.

**Task 2 — `wesad_windows.npz`**
- `X`: shape `(n_windows, n_channels, n_timesteps)` — 60s windows, step 30s, all channels resampled to the same rate (e.g. 4 Hz).
- `y`: `0` = not stressed, `1` = stressed
- `subject`: subject id per window
- `channels`: list of channel names

**Task 3 — `sleepedf_epochs.npz`**
- `X`: shape `(n_epochs, 1, 3000)` — 30s of EEG at 100 Hz
- `y`: `0`=W, `1`=N1, `2`=N2, `3`=N3, `4`=REM
- `subject`, `night`: ids per epoch

### 4.5 Train/Test Splits (important!)

- **Split by person, never by window.** The same person must never be in both train and test, or results look fake-good.
- Use `GroupKFold(n_splits=5)` with `groups=subject`, and a fixed `random_state=42`.
- Save the fold assignment to `processed/folds.json` so every model uses identical splits.

---

## 5. Colab Workflow

**Code is written locally (by humans + agents) → pushed to GitHub → Colab only runs it.** Nobody writes real code inside Colab.

### 5.1 The Runner Notebook (`notebooks/colab_runner.ipynb`)

The notebook should contain only these cells:

```python
# 1. Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Get latest code
!git clone https://github.com/<OWNER>/<REPO>.git /content/repo || (cd /content/repo && git pull)
%cd /content/repo

# 3. Install
!pip install -q -r requirements.txt

# 4. Point code at Drive
import os
os.environ["DATA_DIR"] = "/content/drive/MyDrive/hackachusetts"

# 5. Run ONE job (examples)
!python -m src.run --task hr --model timesfm --quick
!python -m src.run --task hr --model timesfm
```

### 5.2 Colab Survival Rules

1. **Runtime → GPU (T4)** for neural nets and foundation models. CPU is fine for ARIMA/ETS/XGBoost.
2. **Always test with `--quick` first** (2–3 subjects, 1 fold). Only run the full job if quick works.
3. **Save results after every fold** (append to `results.csv`). Colab disconnects; never lose more than one fold.
4. **One job per Colab session.** Split jobs across our 3 accounts to dodge free-tier limits.
5. Foundation model weights download from Hugging Face each session — that's fine, don't store them on Drive.

### 5.3 Space Budget (approx)

| Item | Size | Lives in |
|------|------|----------|
| Raw WESAD + Sleep-EDF subset | ~3–5 GB | Colab temp only |
| Processed data | < 1 GB | Drive |
| Model weights | ~2–3 GB | Colab temp (auto-download) |
| Results, checkpoints | small | Drive |
| Code | tiny | GitHub |

---

## 6. Repo Structure

```
.
├── CLAUDE.md                     # this file
├── .kiro/steering/project.md     # copy of this file for Kiro
├── README.md                     # final public README (results + demo GIF)
├── requirements.txt
├── .gitignore                    # MUST ignore data/, raw/, *.npz, *.parquet, checkpoints/
├── notebooks/
│   └── colab_runner.ipynb
├── src/
│   ├── config.py                 # reads DATA_DIR env var; all paths come from here
│   ├── data/
│   │   ├── wesad.py              # raw WESAD -> processed files
│   │   ├── sleepedf.py           # raw Sleep-EDF -> processed files
│   │   ├── features.py           # hand-made features for XGBoost
│   │   └── splits.py             # builds folds.json
│   ├── models/
│   │   ├── stats_baselines.py    # Naive, ARIMA, ETS
│   │   ├── xgb_baseline.py
│   │   ├── neural_forecast.py    # PatchTST, DeepAR
│   │   ├── neural_classify.py    # small 1D-CNN / LSTM
│   │   ├── timesfm_model.py
│   │   ├── chronos_model.py
│   │   └── moment_model.py       # embeddings + logistic regression
│   ├── eval/
│   │   ├── metrics.py            # MAE, MASE, macro-F1, accuracy
│   │   └── results.py            # append rows to results.csv
│   └── run.py                    # CLI: --task {hr,stress,sleep} --model <name> [--quick]
├── app/
│   └── streamlit_app.py          # demo, reads ONLY results/ and predictions/
└── tests/
    └── test_smoke.py             # tiny checks on 2 subjects
```

**Every model file exposes the same functions** so `run.py` can call any of them:
- Forecasting: `forecast(context_df, horizon) -> predictions_df`
- Classification: `fit(X_train, y_train)` and `predict(X_test) -> labels`

### Results format — `results.csv`
| task | model | tier | fold | metric | value | runtime_s | timestamp |
|------|-------|------|------|--------|-------|-----------|-----------|

---

## 7. Team Roles

| Person | Owns | Folders |
|--------|------|---------|
| **P1 — Data** | Downloading, cleaning, windowing, splits, features | `src/data/`, `tests/` |
| **P2 — Models** | All model wrappers, `run.py`, running jobs on Colab | `src/models/`, `src/run.py`, `src/eval/` |
| **P3 — Demo & Story** | Streamlit app, charts, README, video, Devpost | `app/`, `README.md`, `notebooks/` |

Each person only edits their own folders. Need a change elsewhere? Ask the owner.

---

## 8. Milestones (do in order)

> **Today:** follow the stage order in §0. The milestones below are the checklist; tick them as you go.
> Items marked *(cut today)* are only built if we finish early.

### M0 — Setup
- [x] Repo created, this file committed, `.gitignore` blocks data files
- [x] `requirements.txt` + `src/config.py`
- [ ] `colab_runner.ipynb` clones repo and mounts Drive successfully
- [ ] Team agrees on data formats in §4.4

### M1 — Data (P1)
- [ ] `wesad.py` produces `wesad_hr.parquet` and `wesad_windows.npz`
- [ ] `sleepedf.py` produces `sleepedf_epochs.npz` (20 people)
- [ ] `splits.py` produces `folds.json`
- [ ] `test_smoke.py` passes (shapes, label counts, no subject leaks across folds)
- [ ] Quick plot of one subject's HR + labels to sanity check

### M2 — Baselines (P2)
- [x] `run.py` CLI works end to end with Naive model
- [ ] ARIMA *(cut today)*, ETS on Task 1
- [ ] XGBoost on Tasks 2 and 3
- [ ] First rows in `results.csv` ✅ (we now have something to beat)

### M3 — Foundation Models (P2)
- [ ] TimesFM on Task 1 (zero-shot)
- [ ] Chronos on Task 1 (zero-shot)
- [ ] MOMENT embeddings + logistic regression on Tasks 2 and 3

### M4 — Neural Nets (P2) *(cut today unless ahead of schedule)*
- [ ] PatchTST on Task 1 *(first add-back if ahead by ~5 PM)*
- [ ] DeepAR on Task 1
- [ ] Small 1D-CNN or LSTM on Tasks 2 and 3

### M5 — Demo (P3, can start at M2 using fake/partial results)
- [ ] Leaderboard page (one bar chart per task, grouped by tier)
- [ ] HR forecast page: real HR line + each model's 60s forecast on top
- [ ] Stress page: timeline of true vs predicted stress for one person
- [ ] Sleep page: true vs predicted hypnogram + sleep efficiency score

### M6 — Ship
- [ ] Record demo video (see §10) **as soon as the demo works**
- [ ] README with results table, demo GIF, how to reproduce
- [ ] Devpost submission

### Stretch (only after M6)
- [ ] Fine-tune one small foundation model on Task 1 (short run on Colab)
- [ ] MMASH cortisol task
- [ ] Weights & Biases logging

---

## 9. Agent Rules (Claude Code / Kiro)

1. **One small task per prompt.** Good: "Write `src/data/wesad.py` that loads one subject's wrist data and returns a DataFrame." Bad: "Build the project."
2. **Stay in the owner's folders** listed in §7 for the current human.
3. **Never change the data formats in §4.4 or the function signatures in §6** without the human's OK.
4. **Never commit data**, model weights, or anything in `raw/`, `processed/`, `checkpoints/`.
5. **All paths come from `src/config.py`** (`DATA_DIR` env var). No hard-coded paths.
6. **Every script supports `--quick`** (2–3 subjects, 1 fold) for fast testing.
7. **Check the library's current README** before using TimesFM / Chronos / MOMENT / Nixtla APIs. Don't rely on memory.
8. **Ask before adding a new dependency.** Add it to `requirements.txt` if approved.
9. **Write simple, commented code** — the team are beginners. Short docstring at the top of every function explaining what it does.
10. **After finishing a task**, tick its checkbox in §8 and say what changed in one short summary.
11. Fixed seeds everywhere (`42`).
12. **Respect the stage order and scope cuts in §0.** Don't work on a cut item or a later stage unless the human says so.

---

## 10. Video Plan (2–3 minutes)

1. **Hook (15s):** "Your watch says you're stressed. How good is that guess?"
2. **The idea (20s):** foundation models vs. classic methods vs. neural nets, on real wearable data.
3. **Live demo (60–75s):** HR forecast overlay → stress timeline → sleep hypnogram.
4. **Leaderboard (20s):** one chart, call out the winner per task.
5. **Why it matters (15s):** how a model like this could run on next-gen wearables.
6. **Team + stack (10s).**

---

## 11. Resume Bullet (fill in real numbers at the end)

> Benchmarked 3 time-series foundation models (TimesFM, Chronos, MOMENT) against 7 statistical, ML, and neural baselines on wearable stress, sleep, and heart-rate forecasting using subject-wise cross-validation; zero-shot foundation models improved X by Y% over [best baseline]. Built a Streamlit demo and reproducible Colab pipeline.

---

## 12. Kickoff Prompt (paste into Claude Code or Kiro to start)

```
Read CLAUDE.md fully. I am [P1/P2/P3] and own the folders listed for that role in §7.
We are at stage [1/2/3/4/5] in §0. Look at the tasks for my role in that stage,
propose the single next smallest task, explain it in 2–3 simple sentences, and wait
for my OK before writing code. Follow every rule in §9.
```
