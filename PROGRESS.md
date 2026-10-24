# Build Log — Foundation Models for Wearables (Hackachusetts)

A running record of what we built at each stage, why, and how we checked it works.
Written to help us present the project (video, Devpost, judges' questions).

**Deadline:** 8:00 PM · **Target:** 7:00 PM · **Started:** ~3:00 PM, Sat Oct 24 2026

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
