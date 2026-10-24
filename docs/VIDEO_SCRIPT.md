# Video Script (target 2:30, about 150 words per minute)

**Placeholders to fill (search for `[[`):**
- `[[WINNER_HR]]`, `[[HR_BEST_MAE]]`, `[[HR_BASELINE_MAE]]` heart-rate winner, its MAE, best baseline MAE
- `[[WINNER_STRESS]]`, `[[STRESS_BEST_F1]]` stress winner and macro-F1
- `[[WINNER_SLEEP]]`, `[[SLEEP_BEST_F1]]` sleep winner and macro-F1
- `[[SLEEP_EFFICIENCY]]` sleep efficiency shown for the chosen night (%)
- `[[HEADLINE_FINDING]]` one sentence verdict
- `[[SUBJECT_ID]]` the person you pick in the demo
- `[[TEAM_NAMES]]` names

---

| # | Time | ON SCREEN | WE SAY |
|---|------|-----------|--------|
| 1. Hook | 0:00-0:15 | Title slide: "Can AI that never saw wearable data beat the classics?" Then a smartwatch photo. | "Your watch says you're stressed. How good is that guess? Every day, wearables record our heart, sweat and sleep. But is the AI reading that data actually any good?" |
| 2. Idea | 0:15-0:35 | Simple slide: four boxes, Simple / Classic ML / Neural net / Foundation model. Highlight the last one. | "We tested a new kind of model, time-series foundation models: Chronos, TimesFM and MOMENT. They were trained on lots of other data and never saw ours. We pitted them against simple methods, XGBoost, and a neural net trained on the data. Every model is tested on people it has never seen. The foundation models stay frozen, so it's a fair fight." |
| 3a. Demo: HR | 0:35-1:00 | Run Streamlit. Click sidebar **"HR forecast"**. Pick subject `[[SUBJECT_ID]]`. Show true line vs. model overlays; toggle on [[WINNER_HR]] and Naive. | "First task: forecast the next 60 seconds of heart rate. The black line is reality. Here's the naive guess, which just repeats the last value. And here's [[WINNER_HR]], a zero-shot model. It follows the real curve closely: [[HR_BEST_MAE]] beats per minute of error, against [[HR_BASELINE_MAE]] for the best simple baseline." |
| 3b. Demo: Stress | 1:00-1:25 | Click **"Stress"**. Same subject. Point to the shaded true-stress period, then the predicted timeline under it. | "Second task: stress detection. This is a stress test from the WESAD study. The top bar is the truth, the bottom is the model's call, one guess per minute. It lights up when the person is under pressure. Best model: [[WINNER_STRESS]], [[STRESS_BEST_F1]] macro-F1." |
| 3c. Demo: Sleep | 1:25-1:50 | Click **"Sleep"**. Show true vs. predicted hypnogram. Point at the sleep efficiency number. | "Third task: sleep from brainwaves. This staircase is a full night, Wake, REM, light and deep sleep, every 30 seconds. Top is what a sleep lab says, bottom is our model, [[WINNER_SLEEP]], at [[SLEEP_BEST_F1]] macro-F1. From that we get a sleep score: this night's efficiency was [[SLEEP_EFFICIENCY]] percent." |
| 4. Leaderboard | 1:50-2:10 | Click **"Leaderboard"**. Show the three bar charts. Circle the winner in each. | "Here's the full scoreboard, colored by model type. [[HEADLINE_FINDING]] Winners: [[WINNER_HR]] for heart rate, [[WINNER_STRESS]] for stress, [[WINNER_SLEEP]] for sleep." |
| 5. Why it matters | 2:10-2:25 | Back to watch photo, or slide: "On-device, no per-user training." | "Why it matters: a model that works without training on you could run on your next watch, giving useful health insights from day one, with data staying on your wrist." |
| 6. Team + stack | 2:25-2:35 | Slide: [[TEAM_NAMES]], logos/list: PyTorch, Hugging Face, XGBoost, Streamlit, Colab. | "We're [[TEAM_NAMES]]. Built with PyTorch, Hugging Face, XGBoost, Streamlit and Colab. Thanks for watching." |

Timing note: if over 3:00, cut the sentence about the naive guess in 3a and the stack list in 6.

## Recording checklist
- [ ] Fill every `[[...]]` above from the real `results.csv` before recording. Never read out numbers from fake or test data.
- [ ] Confirm the demo shows REAL results: no "fake data" warning banner in the app.
- [ ] `streamlit run app/streamlit_app.py` from the repo root.
- [ ] Pick a subject with a clear stress period and a model forecast visibly better than Naive. Use the same subject on HR and Stress pages.
- [ ] For Sleep, pick a night with all five stages and a good-looking prediction.
- [ ] Sidebar pages in order: Leaderboard, HR forecast, Stress, Sleep. Rehearse clicking in demo order (HR, Stress, Sleep, then Leaderboard).
- [ ] Browser zoom 125%, hide bookmarks, close notifications, full-screen window.
- [ ] Do a dry run with a timer; keep the total under 3:00.
- [ ] Record screen and voice in one take per segment; record audio in a quiet room.
- [ ] Upload, set link to public/unlisted, and paste into Devpost.
