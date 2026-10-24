# Build Log — Foundation Models for Wearables (Hackachusetts)

A running record of what we built at each stage, why, and how we checked it works.
Written to help us present the project (video, Devpost, judges' questions).

**Deadline:** 8:00 PM · **Target:** 7:00 PM · **Started:** ~3:00 PM, Sat Oct 24 2026

## 📍 Current status (updated every stage)

| Stage | Status | Finished |
|---|---|---|
| 1. Setup | ✅ Done | ~3:05 PM |
| 2. Data + first results (Naive, ETS) | ✅ Done | ~3:15 PM |
| 3. Chronos, TimesFM, XGBoost, Sleep-EDF | ✅ Done | ~4:15 PM |
| 4. MOMENT + demo stress/sleep pages | ⏳ Next | target 6:20 PM |
| 4b. PatchTST (add-back, we're ahead) | ⏳ After 4 | — |
| 5. Ship: video, README, Devpost | ⏳ | target 7:00 PM |

**Real data so far:** Sleep-EDF ✅ (tested on 3 real people) · WESAD ❌ (waiting on download link)
**Real jobs on Colab:** none run yet. All scores below are from local tests unless marked REAL.

---

## Stage 1 — Setup (done ~3:05 PM)

**Goal:** an empty-but-working project that all 3 of us + Colab can use.

**What we built**
- `CLAUDE.md`: the full project plan, split into time-boxed stages (§0) with scope cuts so we finish on time.
  `.kiro/steering/project.md` links to it so Kiro uses the same plan.
- `src/config.py`: every file path comes from ONE place (the `DATA_DIR` setting).
  On Colab that's Google Drive; locally it's `./data`. Raw downloads go to Colab's temporary disk, never Drive.
- `.gitignore`: blocks all data files (`*.npz`, `*.parquet`, `*.edf`, `*.csv`, model weights) so we never push data to GitHub.
- `requirements.txt`: core packages only. Foundation-model packages (Chronos, TimesFM, MOMENT) are
  installed one per Colab session because they are big and can conflict with each other.
- `notebooks/colab_runner.ipynb`: Colab only *runs* code from GitHub: mount Drive → clone repo → install →
  download data → run one job.

**Decisions + why**
- **Cut scope** to fit ~4 hours: dropped ARIMA, DeepAR, CNN/LSTM and the stretch goals. PatchTST is the first thing we add back if we're ahead.
- **Code is written locally, Colab only runs it**, so nobody loses work when Colab disconnects.

**How we checked it**
- 9 automated tests: paths follow `DATA_DIR`, file names match the plan, raw data never goes to Drive,
  and `.gitignore` blocks data but not code.
- **Bug caught by testing:** the rule `data/` in `.gitignore` was also hiding our code folder `src/data/`.
  We fixed it (`/data/` = only the top-level folder) and added a test so it can't come back.

**Team:** invited Rith-2056 to the GitHub repo (still waiting on the 3rd teammate's GitHub username).

---

## Stage 2 — Data + first results (done ~3:15 PM, ahead of the 4:30 target)

**Goal:** turn raw sensor recordings into clean tables, and get a first score to beat.

**What we built**
- **Data (P1)**: `src/data/wesad.py`
  - Reads each WESAD person's wrist recording (pulse/BVP, sweat/EDA, skin temperature, movement/ACC).
  - **Task 1 file:** heart rate, one value per second → `wesad_hr.parquet`.
  - **Task 2 file:** 60-second chunks (every 30 s) of all 6 wrist channels, labelled calm (0) or stressed (1)
    → `wesad_windows.npz`. Only chunks where the person was doing ONE thing the whole time are kept.
- **Splits (P1)**: `src/data/splits.py` puts each *person* into one of 5 groups ("folds").
  A model is always tested on people it has never seen, so results aren't fake-good.
- **Scoring + runner (P2)**: `src/eval/metrics.py` (MAE, MASE, F1, accuracy), `src/eval/results.py`
  (saves a score after every fold, so a Colab crash loses at most one fold), and `src/run.py`, one
  command to run any model: `python -m src.run --task hr --model ets`.
- **First models (P2)**: `src/models/stats_baselines.py`
  - **Naive** guesses "heart rate stays the same as right now."
  - **ETS** (exponential smoothing) continues the recent trend.
  - These are the "simple tier" every fancy model has to beat.
- **Demo (P3)**: `app/streamlit_app.py`, a website with a leaderboard (bar chart per task, colored by model
  type) and a heart-rate forecast viewer. It uses fake numbers until real results exist, with a warning banner.

**Decisions + why**
- **Heart rate from the pulse signal (BVP), not the watch's own HR file:** the pulse signal is already
  lined up in time with the stress labels; the HR file uses a different clock.
- **8 Hz instead of 4 Hz for stress chunks:** at 4 Hz the pulse wave gets destroyed. 60 s × 8 Hz = 480
  points, which also fits MOMENT's input size (512).
- **`--quick` runs are saved separately** (`results_quick.csv`) so test runs never appear on the leaderboard.

**How we checked it**
- Test-driven: wrote tests first, then code. 29 tests pass.
- Built a **fake WESAD dataset** with the exact same file structure as the real one (simulated 70 bpm pulse)
  so we could test everything before the real download finished.
- End-to-end smoke test on 5 fake people: process data → make folds → run Naive and ETS (quick + full)
  → 20 score rows saved → demo app read them with no errors.
- Fake-data numbers (just a sanity check, NOT real results): ETS MAE 5.17 bpm vs Naive 5.33 bpm.

**Still to do with real data:** run `python -m src.data.wesad --quick` on Colab with the real WESAD download.

---

## Stage 3 — Foundation models + stress + sleep data (done ~4:15 PM, target was 5:30)

**Goal:** get the big pre-trained AI models forecasting heart rate, a classic ML model detecting stress,
and the sleep dataset ready.

**What we built**
- **Chronos-2 (Amazon)**: `src/models/chronos_model.py`. Zero-shot heart-rate forecasting: it gets the
  last 512 s of heart rate and predicts the next 60 s. It has **never been trained on our data**.
- **TimesFM 3.0 (Google)**: `src/models/timesfm_model.py`. Same job, same inputs and outputs, different company's model.
- **Hand-made features (P1)**: `src/data/features.py` turns each raw chunk into summary numbers:
  - Stress: average/spread/min/max/percentiles/trend of each wrist channel, plus overall movement.
  - Sleep: the same stats plus EEG "band powers" (how much slow delta waves vs. fast beta waves). This is
    how sleep doctors read EEG.
- **XGBoost (P2)**: `src/models/xgb_baseline.py` is the classic-ML tier. It learns from those features.
  It is trained only on people in the *other* folds, and balances classes so rare sleep stages (N1) count.
- **Sleep-EDF processing (P1)**: `src/data/sleepedf.py` takes brainwaves (Fpz-Cz EEG) → 30 s chunks →
  labels W/N1/N2/N3/REM. Stages 3+4 are merged into N3 (modern standard). Long awake periods are trimmed to
  30 min before and after sleep (standard practice; otherwise ~70% of the data is "awake").
- **run.py** now runs stress and sleep too: `python -m src.run --task stress --model xgboost`.
  It saves predictions with timestamps / epoch numbers so the demo can draw timelines and hypnograms.
- **One shared rule for classifiers:** every model file has `build(task)` → `.fit(X, y)` / `.predict(X)`,
  so adding MOMENT in Stage 4 is plug-and-play.

**Decisions + why**
- **Newest model versions** (the plan says to use the newest): Chronos-2 (120M parameters, Oct 2025) and
  TimesFM 3.0 (330M). We read each library's current README first, because TimesFM 3.0 changed its whole
  API (`timesfm3` package). Code written from memory would have broken.
- **Foundation-model packages are installed one per Colab session**; we also checked locally that
  Chronos + TimesFM install together without conflicts.

**How we checked it**
- 42+ automated tests. The foundation-model tests download the **real model weights** and run them.
- Fake-data sanity numbers (NOT real results), HR forecast MAE: Chronos 1.47, TimesFM 1.46, vs.
  ETS ~5.2 and Naive ~5.3 bpm. Each took ~7 s for 18 windows on a laptop CPU.
- XGBoost on fake stress data scored ~50% (coin flip). That's **what should happen**: the fake data has
  no real stress signal. A high score would have meant a bug leaking test answers into training.
- **Tested on REAL sleep data:** downloaded 3 real Sleep-EDF people (6 nights) and ran XGBoost.
  - 841–1,186 thirty-second chunks per night (≈ 7–9 h). EEG ±200 µV. All 5 stages present; N2 is the most common, as in real sleep.
  - **First real result:** XGBoost sleep staging macro-F1 **0.44 / 0.67 / 0.72** on the 3 folds (accuracy 57–76%).
    Only 2 people to learn from per fold, so we expect this to go up with all 20.

**Problems we hit and fixed**
- **PhysioNet's download server stalled** (~0 KB/s). We switched to PhysioNet's official AWS Open Data mirror
  (same files, checksums still verified): 2 nights now download in **8 seconds**.
- **The fold builder crashed with fewer than 5 people** (e.g. after a `--quick` preprocess). It now uses
  fewer folds automatically; we added a test for it. Found by running the real pipeline, not by the unit tests.
- **The test suite froze on macOS**: XGBoost and PyTorch each ship their own OpenMP threading library and can
  deadlock in one Python process. The real jobs are never affected (one model per process, Colab is Linux);
  the foundation-model tests now run each model in its own process. Full suite: 43 tests in ~24 s.
