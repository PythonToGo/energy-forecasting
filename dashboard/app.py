"""Streamlit dashboard: recent actual consumption + recursive forward forecast."""

import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
DATA_PATH = os.environ.get("PROCESSED_DATA_PATH", "data/processed/energy_clean.csv")

st.set_page_config(page_title="Energy Forecast", page_icon="🔋", layout="wide")
st.title("⚡ Home Energy Consumption Forecast")
st.caption("Recursive multi-step forecasting from recent consumption (lags + rolling).")


# ── Sidebar: model info ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model Info")
    try:
        info = requests.get(f"{API_URL}/model/info", timeout=3).json()
        st.metric("Test MAE", f"{info.get('test_mae', '—')} kW")
        st.metric("Test RMSE", f"{info.get('test_rmse', '—')} kW")
        skill = info.get("skill_vs_daily_baseline")
        st.metric("Skill vs daily baseline", f"{skill:.1%}" if isinstance(skill, float) else "—")
        with st.expander("Baselines"):
            st.write(f"Naive last-hour MAE: **{info.get('baseline_naive_last_mae', '—')}**")
            st.write(f"Naive daily MAE: **{info.get('baseline_naive_daily_mae', '—')}**")
            st.write(f"Naive weekly MAE: **{info.get('baseline_naive_weekly_mae', '—')}**")
        with st.expander("Training details"):
            st.write(
                f"Train / Val / Test: {info.get('train_size', '—')} / "
                f"{info.get('val_size', '—')} / {info.get('test_size', '—')} hours"
            )
            st.write(f"Features ({len(info.get('features', []))}): {info.get('features', '—')}")
    except Exception:
        st.warning("API not reachable — start the server first.")


# ── Forecast ──────────────────────────────────────────────────────────────────
st.header("Forecast")
horizon = st.slider("Horizon (hours)", min_value=1, max_value=168, value=24)

if st.button("Run forecast", type="primary"):
    with st.spinner("Forecasting…"):
        try:
            resp = requests.get(f"{API_URL}/forecast", params={"horizon": horizon}, timeout=30)
            if resp.status_code == 200:
                payload = resp.json()
                fc = pd.DataFrame(payload["forecast"])
                fc["timestamp"] = pd.to_datetime(fc["timestamp"])
                st.success(f"Forecast from {payload['from_timestamp']} for {horizon}h")
                band = (
                    alt.Chart(fc)
                    .mark_area(opacity=0.25, color="#1f77b4")
                    .encode(x="timestamp:T", y=alt.Y("p10:Q", title="kW"), y2="p90:Q")
                )
                median = alt.Chart(fc).mark_line(color="#1f77b4").encode(x="timestamp:T", y="p50:Q")
                st.caption("Line = P50 (median); shaded band = P10–P90 (80% interval)")
                st.altair_chart(band + median, use_container_width=True)
                with st.expander("Forecast data"):
                    st.dataframe(fc)
            else:
                st.error(f"API error {resp.status_code}: {resp.text}")
        except Exception as e:
            st.error(f"Could not reach API: {e}")


# ── Recent actual consumption ────────────────────────────────────────────────
st.divider()
st.header("Recent Actual Consumption (last 3 days)")
try:
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"], index_col="datetime")
    cutoff = df.index.max() - pd.Timedelta(days=3)
    st.line_chart(df.loc[df.index >= cutoff, "Global_active_power"], height=220)
    st.caption(
        f"Data range: {df.index.min().date()} → {df.index.max().date()} "
        f"| {len(df):,} hourly records"
    )
except Exception as e:
    st.error(f"Error loading data: {e}")
