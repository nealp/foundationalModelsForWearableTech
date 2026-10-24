# Devpost Submission Draft

*Numbers below are real (subject-wise cross-validation). Heart-rate and stress pipelines are built and tested but were not run on WESAD in time.*

---

## Inspiration
Your watch says you're stressed. How good is that guess? Wearables collect heart rate, sweat, skin temperature and brainwaves all day, but most analysis is hand-tuned per task. A new family of time-series foundation models (Chronos-2, TimesFM 3.0, MOMENT) was pre-trained on huge piles of other data. We asked: can a model that has never seen wearable data beat the classic methods, and the neural nets trained directly on it?

## What it does
A benchmark plus a live demo. Three tasks:
1. **Heart-rate forecast:** predict the next 60 seconds (WESAD wrist data).
2. **Stress detection:** stressed vs. calm per 60-second window (WESAD).
3. **Sleep staging:** Wake / N1 / N2 / N3 / REM per 30 seconds from EEG (Sleep-EDF), plus a sleep efficiency score per night.

Our Streamlit app has a leaderboard, a forecast overlay, a stress timeline and a sleep hypnogram. **Headline: a frozen foundation model that never saw a brainwave matches hand-engineered sleep features.** On sleep staging (20 people, 42,308 thirty-second EEG epochs), MOMENT + a linear probe scored **0.696 macro-F1 (76.9% accuracy)** vs. XGBoost's **0.705 (77.5%)** on the same unseen people. XGBoost averages **0.69 ± 0.03** over all 5 folds. With only 2 people to learn from, MOMENT **won clearly: 0.635 vs. 0.435**. Foundation models shine when labelled data is scarce, which is exactly the wearable situation. On a real night, true sleep efficiency was 77.6% and predicted 78.1%. The heart-rate (Naive, ETS, PatchTST, Chronos-2, TimesFM 3.0) and stress (XGBoost, MOMENT) pipelines are fully built and tested end to end; we ran out of time to run them on WESAD.

## How we built it
We built 9 models in four tiers:
- **Simple:** Naive (repeat last value) and ETS.
- **Classic ML:** XGBoost on hand-made features (including EEG band powers for sleep).
- **Neural net:** PatchTST, trained from scratch on our data.
- **Foundation models:** Chronos-2 and TimesFM 3.0 (zero-shot forecasting), MOMENT (classification).

**Fair testing:** we use subject-wise cross-validation. People are split into 5 folds, and every model is scored on people it never saw. Data: 20 Sleep-EDF people (39 nights); WESAD (15 people) pipeline ready.

**Fair to the foundation models too:** they stay frozen. Chronos-2 and TimesFM forecast with zero training on our data. For MOMENT, we extract embeddings (a numeric fingerprint of each window) and train only a tiny logistic regression on top, a "linear probe." The big model is never fine-tuned. The other tiers do train on our data, so any foundation-model win is not from extra training.

A single command (`python -m src.run --task hr --model ets`) runs any model, saves scores after every fold, and a Colab notebook runs everything from GitHub.

## Challenges we ran into
- **PhysioNet's server stalled** at ~0 KB/s. We switched to PhysioNet's official AWS mirror (checksums still verified): two nights now download in 8 seconds.
- **TimesFM 3.0 changed its whole API** (new `timesfm3` package). We read the current README instead of coding from memory.
- **MOMENT pins numpy 1.25,** which won't install on Python 3.12 (Colab). We install it with `--no-deps` and verified it works with current libraries.
- **A macOS deadlock in tests:** torch and XGBoost each ship their own OpenMP library. We now run each model test in its own process.
- **`.gitignore` was hiding `src/data/`** (the rule `data/` matched our code folder). Fixed with `/data/` and added a test.
- **The fold builder crashed with fewer than 5 people.** It now uses fewer folds automatically.

## Accomplishments that we're proud of
- A clean, leak-free comparison across four model tiers on three tasks.
- Real foundation-model weights running in tests, not mocks.
- A reproducible pipeline: one command per model, crash-safe scoring, and a demo that tells you when it is showing placeholder data.
- Built in one afternoon with three people.

## What we learned
- Evaluation design matters more than model choice. Testing on unseen people is what keeps scores honest, and a fake-data XGBoost scoring about 50% (a coin flip) was the sanity check that told us there was no leakage.
- Squashing a 30-second EEG chunk to MOMENT's 512 steps would erase fast brainwaves, so we cut it into 6 pieces instead.
- Library APIs for new models move fast. Read the README first.

## What's next
- **Fine-tuning** the foundation models, to see how much frozen vs. tuned matters.
- **On-device wearables:** small models running on the watch itself.
- **MMASH cortisol prediction:** a hormone-level stress signal beyond a yes/no label.
- All 20 Sleep-EDF people and more seeds.

## Built with
Python, PyTorch, Hugging Face, Chronos-2 (Amazon), TimesFM 3.0 (Google), MOMENT (CMU), PatchTST, statsforecast, neuralforecast, XGBoost, scikit-learn, MNE, pandas, PyArrow, NumPy, Matplotlib, Streamlit, Google Colab, GitHub. Datasets: WESAD, Sleep-EDF (PhysioNet).
