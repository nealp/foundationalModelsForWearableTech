"""Demo app: benchmark of time-series foundation models on wearable data.

Run from the repo root:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

# Make `src` and `app` importable when launched from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from app import data
from src import config

TASK_TITLES = {"hr": "Heart-rate forecasting (MAE, lower is better)",
               "stress": "Stress detection (macro F1, higher is better)",
               "sleep": "Sleep staging (macro F1, higher is better)"}
TIER_COLORS = {1: "#9aa5b1", 2: "#4c9be8", 3: "#f2a33a", 4: "#d9534f"}


def plot_task(table: pd.DataFrame, task: str):
    """Draw a bar chart (mean with std error bars, colored by tier) for one task."""
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = [TIER_COLORS[t] for t in table["tier"]]
    ax.bar(table["model"], table["mean"], yerr=table["std"], color=colors, capsize=4)
    ax.set_ylabel(data.MAIN_METRIC[task])
    # Legend: only tiers that appear in this chart
    for tier in sorted(table["tier"].unique()):
        ax.bar(0, 0, color=TIER_COLORS[tier], label=data.TIER_NAMES[tier])
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def leaderboard_page():
    """Show one bar chart and one table per task."""
    st.title("Leaderboard")
    df, is_fake = data.load_results(config.RESULTS_CSV)
    if is_fake:
        st.warning("FAKE DATA: no results.csv found, so these numbers are made up for layout only.")
    for task in ["hr", "stress", "sleep"]:
        st.subheader(TASK_TITLES[task])
        table = data.leaderboard(df, task)
        if table.empty:
            st.info("No results for this task yet.")
            continue
        st.pyplot(plot_task(table, task))
        st.dataframe(table.round(3), hide_index=True)


def hr_page():
    """Plot the true HR and each model's forecast for a chosen subject and window."""
    st.title("HR forecast")
    # skip *_quick files: those come from --quick test runs, not real results
    files = [f for f in sorted(Path(config.PREDICTIONS_DIR).glob("hr_*.parquet")) if not f.stem.endswith("_quick")]
    if not files:
        st.info("No HR prediction files found yet (expected hr_<model>.parquet in the predictions folder).")
        return
    preds = {f.stem[len("hr_"):]: pd.read_parquet(f) for f in files}
    all_df = pd.concat(preds.values())
    subject = st.selectbox("Subject", sorted(all_df["unique_id"].unique()))
    starts = sorted(all_df.loc[all_df["unique_id"] == subject, "window_start"].unique())
    start = st.selectbox("Window start (s)", starts)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    first = True
    for model, df in preds.items():
        w = df[(df["unique_id"] == subject) & (df["window_start"] == start)].sort_values("ds")
        if first and not w.empty:
            ax.plot(w["ds"], w["y"], color="black", lw=2, label="true HR")
            first = False
        ax.plot(w["ds"], w["y_hat"], "--", label=model)
    ax.set_ylabel("HR (bpm)")
    ax.legend(fontsize=7)
    fig.autofmt_xdate()
    st.pyplot(fig)


def _load_preds(task: str):
    """Real predictions if present, else fake ones with a warning banner. None if nothing at all."""
    preds = data.load_predictions(task)
    if not preds:
        preds = data.make_fake_predictions(task)
        st.warning("FAKE DATA: no real prediction files found, so these curves are made up for layout only.")
    return preds


def stress_page():
    """Timeline of true vs predicted stress for one subject."""
    st.title("Stress detection")
    preds = _load_preds("stress")
    model = st.selectbox("Model", sorted(preds))
    df = preds[model]
    subject = st.selectbox("Subject", sorted(df["subject"].unique()))
    d = df[df["subject"] == subject].sort_values("start_s")
    if d.empty:
        st.info("No windows for this subject.")
        return
    t = d["start_s"].to_numpy() / 60.0
    fig, axes = plt.subplots(2, 1, figsize=(8, 2.8), sharex=True)
    for ax, col, name, color in [(axes[0], "y", "True", "black"), (axes[1], "y_pred", "Predicted", "#d9534f")]:
        ax.step(t, d[col], where="post", color=color)
        ax.fill_between(t, 0, d[col], step="post", color=color, alpha=0.25)
        ax.set_yticks([0, 1], ["calm", "stress"])
        ax.set_ylabel(name)
    axes[1].set_xlabel("Time (minutes)")
    fig.tight_layout()
    st.pyplot(fig)
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{(d['y'] == d['y_pred']).mean():.1%}")
    c2.metric("Truly stressed", f"{d['y'].mean():.1%} of time")
    c3.metric("Predicted stressed", f"{d['y_pred'].mean():.1%} of time")


def sleep_page():
    """Hypnograms of true vs predicted sleep stages with sleep efficiency."""
    st.title("Sleep staging")
    preds = _load_preds("sleep")
    model = st.selectbox("Model", sorted(preds))
    df = preds[model]
    subject = st.selectbox("Subject", sorted(df["subject"].unique()))
    sub = df[df["subject"] == subject]
    night = st.selectbox("Night", sorted(sub["night"].unique()))
    d = sub[sub["night"] == night].sort_values("epoch")
    if d.empty:
        st.info("No epochs for this night.")
        return
    hours = d["epoch"].to_numpy() * 30 / 3600.0
    fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
    for ax, col, name, color in [(axes[0], "y", "True", "black"), (axes[1], "y_pred", "Predicted", "#d9534f")]:
        ax.step(hours, data.hypnogram_order(d[col]), where="post", color=color, lw=1)
        ax.set_yticks(list(data.HYP_TICKS), list(data.HYP_TICKS.values()))
        ax.set_ylabel(name)
    axes[1].set_xlabel("Time (hours)")
    fig.tight_layout()
    st.pyplot(fig)
    c1, c2, c3 = st.columns(3)
    c1.metric("Sleep efficiency (true)", f"{data.sleep_efficiency(d['y']):.1%}")
    c2.metric("Sleep efficiency (predicted)", f"{data.sleep_efficiency(d['y_pred']):.1%}")
    c3.metric("Night accuracy", f"{(d['y'] == d['y_pred']).mean():.1%}")


def main():
    """Draw the sidebar navigation and the selected page."""
    page = st.sidebar.radio("Page", ["Leaderboard", "HR forecast", "Stress", "Sleep"])
    if page == "Leaderboard":
        leaderboard_page()
    elif page == "HR forecast":
        hr_page()
    elif page == "Stress":
        stress_page()
    else:
        sleep_page()


main()
