# Foundation Models for Wearables

> **"Your watch says you're stressed. How good is that guess — and can a model that has never seen wearable data do better?"**

We benchmark new **time-series foundation models** (Google **TimesFM 3.0**, Amazon **Chronos-2**, CMU **MOMENT**),
used *out of the box*, against classic statistics, classic machine learning, and neural networks trained directly
on real wearable data. Built at **Hackachusetts 2026**.

## The 3 tasks

| Task | Input | Output | Dataset | Metric |
|---|---|---|---|---|
| ❤️ **Heart-rate forecast** | last 512 s of heart rate (wrist pulse) | next 60 s | WESAD (Empatica E4 wrist) | MAE, bpm (lower = better) |
| 😰 **Stress detection** | 60 s of wrist pulse, sweat, skin temp, movement | calm / stressed | WESAD | macro-F1 (higher = better) |
| 😴 **Sleep staging** | 30 s of brainwaves (EEG Fpz-Cz) | Wake / N1 / N2 / N3 / REM | Sleep-EDF (20 people) | macro-F1 (higher = better) |

Sleep staging also gives a per-night **sleep efficiency** (time asleep ÷ time in bed) in the demo.

## The 4 model tiers

| Tier | Models | Trained on our data? |
|---|---|---|
| 1 · Simple | Naive (repeat last value), ETS | fitted per window |
| 2 · Classic ML | XGBoost on hand-made features (signal stats, EEG band powers) | yes |
| 3 · Neural net | PatchTST (heart rate) | yes |
| 4 · Foundation | **TimesFM 3.0, Chronos-2** (heart rate, zero-shot) · **MOMENT-1-base** (stress, sleep) | **no — frozen** |

For classification, MOMENT stays frozen: it turns each window into an embedding, and only a small logistic
regression is trained on top (a "linear probe").

**Fair testing:** every score uses **subject-wise 5-fold cross-validation**. A model is always tested on people it
has never seen, so it can't memorize anyone.

## Results

<!-- Generated with: python -m src.eval.summary -->
RESULTS_TABLE

## Demo

```bash
streamlit run app/streamlit_app.py
```
Pages: **Leaderboard** · **HR forecast** (real heart rate + each model's 60 s forecast) · **Stress** (true vs.
predicted timeline per person) · **Sleep** (true vs. predicted hypnogram + sleep efficiency).

## Reproduce

Code is written locally and **run on Google Colab** (`notebooks/colab_runner.ipynb`). Raw data never touches
Drive or GitHub; only small processed files are saved.

```bash
pip install -r requirements.txt
python -m src.data.wesad          # raw WESAD -> processed/wesad_hr.parquet + wesad_windows.npz
python -m src.data.sleepedf       # downloads Sleep-EDF (AWS mirror) -> processed/sleepedf_epochs.npz
python -m src.data.splits         # subject-wise folds -> processed/folds.json

# one job per run; add --quick for a 3-person smoke test first
python -m src.run --task hr     --model naive|ets|chronos|timesfm|patchtst
python -m src.run --task stress --model xgboost|moment
python -m src.run --task sleep  --model xgboost|moment
python -m src.eval.summary        # leaderboard table
```

Foundation-model packages are installed per job (they pin conflicting versions):
`pip install chronos-forecasting` · `pip install "timesfm[torch]"` · `pip install --no-deps momentfm` · `pip install neuralforecast`.
Set `DATA_DIR` to choose where processed data and results go (default `./data`).

Tests: `python -m pytest -q tests` (fake datasets shaped like the real ones, plus real-weights checks for each foundation model).

## Repo map

```
src/data/     WESAD + Sleep-EDF preprocessing, features, subject-wise folds
src/models/   stats_baselines, xgb_baseline, neural_forecast (PatchTST), chronos_model, timesfm_model, moment_model
src/eval/     metrics, per-fold results writer, leaderboard summary
src/run.py    one CLI for every task x model
app/          Streamlit demo
PROGRESS.md   build log: what we built at each stage, decisions, and bugs we fixed
```

## Datasets

- **WESAD**: Schmidt et al., *Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection*, ICMI 2018.
- **Sleep-EDF Expanded** (sleep-cassette): Kemp et al., 2000; via PhysioNet (Goldberger et al., 2000).
