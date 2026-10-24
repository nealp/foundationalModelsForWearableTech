# Video Script (target 2:30, about 150 words per minute)


---

| # | Time | ON SCREEN | WE SAY |
|---|------|-----------|--------|
| 1. Hook | 0:00-0:15 | Title slide: "Can AI that never saw wearable data beat the classics?" Then a smartwatch photo. | "Your watch says you're stressed. How good is that guess? Every day, wearables record our heart, sweat and sleep. But is the AI reading that data actually any good?" |
| 2. Idea | 0:15-0:35 | Simple slide: four boxes, Simple / Classic ML / Neural net / Foundation model. Highlight the last one. | "We tested a new kind of model, time-series foundation models: Chronos, TimesFM and MOMENT. They were trained on lots of other data and never saw ours. We pitted them against simple methods, XGBoost, and a neural net trained on the data. Every model is tested on people it has never seen. The foundation models stay frozen, so it's a fair fight." |
| 3a. The pipeline | 0:35-0:55 | Show the README tiers table (or the HR forecast page with its FAKE DATA banner visible). | "We built nine models in four tiers, from just repeating the last heart-rate value up to Google's TimesFM and Amazon's Chronos, plus heart-rate forecasting and stress detection pipelines on wrist data. All of it is built and tested end to end. Tonight we ran the full benchmark on sleep." |
| 3b. Demo: Sleep | 0:55-1:50 | Click **"Sleep"**. Model **xgboost**, subject **SC0**, night **1**. Point at true vs. predicted hypnogram, then the efficiency metrics. Switch model to **moment** if available. | "This is a real night of sleep from brainwaves, Wake, REM, light and deep sleep, every 30 seconds. Top is what a sleep doctor scored, bottom is our model, tested on a person it never saw. Sleep efficiency: 77.6 percent true, 78.1 percent predicted. Now the big question: what about a foundation model that has never seen a single brainwave? MOMENT, completely frozen, with just a tiny linear layer on top, scores 0.70 F1, matching hand-engineered sleep features at 0.71." |
| 4. Leaderboard | 1:50-2:10 | Click **"Leaderboard"**. Point at the Sleep chart. | "Here's the scoreboard. Across 20 people and 42 thousand sleep epochs, it's a tie at full data. But give each model only two people to learn from, and the foundation model wins by a mile: 0.64 versus 0.44. That's the wearable reality: lots of signal, very few labels." |
| 5. Why it matters | 2:10-2:25 | Back to watch photo, or slide: "On-device, no per-user training." | "Why it matters: a model that works without training on you could run on your next watch, giving useful health insights from day one, with data staying on your wrist." |
| 6. Team + stack | 2:25-2:35 | Slide: [TEAM NAMES], logos/list: PyTorch, Hugging Face, XGBoost, Streamlit, Colab. | "We're [TEAM NAMES]. Built with PyTorch, Hugging Face, XGBoost, Streamlit and Colab. Thanks for watching." |

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
