import streamlit as st
import requests
import datetime
import pandas as pd

st.set_page_config(page_title="Energy Prediction", page_icon="🔋", layout="wide")

st.title("⚡  Home Energy Consumption Forecast Dashboard")
st.subheader("by Taey👩‍💻")

API_URL = "http://localhost:8000"


# ── Sidebar: model info ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model Info")
    try:
        info = requests.get(f"{API_URL}/model/info", timeout=3).json()
        st.metric("Test MAE", f"{info.get('test_mae', '—')} kW")
        st.metric("Test RMSE", f"{info.get('test_rmse', '—')} kW")
        st.metric("Skill vs Daily Baseline", f"{info.get('skill_vs_daily_baseline', '—'):.1%}" if isinstance(info.get('skill_vs_daily_baseline'), float) else "—")
        with st.expander("Baseline comparison"):
            st.write(f"Naive last-hour MAE: **{info.get('baseline_naive_last_mae', '—')}**")
            st.write(f"Naive daily MAE: **{info.get('baseline_naive_daily_mae', '—')}**")
            st.write(f"Naive weekly MAE: **{info.get('baseline_naive_weekly_mae', '—')}**")
        with st.expander("Training details"):
            st.write(f"Trained at: {info.get('trained_at', '—')}")
            st.write(f"Train / Val / Test: {info.get('train_size', '—')} / {info.get('val_size', '—')} / {info.get('test_size', '—')} hours")
            st.write(f"Val MAE: {info.get('val_mae', '—')} kW")
            st.write(f"Features: {info.get('features', '—')}")
    except Exception:
        st.warning("API not reachable — start the server first.")


# ── Main: single prediction ──────────────────────────────────────────────────
st.header("Single-point Prediction")
st.write("Predict energy consumption for a specific hour, day, and month.")

col1, col2, col3 = st.columns(3)
with col1:
    hour = st.number_input("Hour (0–23)", min_value=0, max_value=23, value=datetime.datetime.now().hour)
with col2:
    dayofweek = st.selectbox(
        "Day of Week",
        options=list(range(7)),
        format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x],
        index=datetime.datetime.now().weekday(),
    )
with col3:
    month = st.selectbox(
        "Month",
        options=list(range(1, 13)),
        format_func=lambda x: ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][x-1],
        index=datetime.datetime.now().month - 1,
    )

if "history" not in st.session_state:
    st.session_state.history = []

if st.button("Predict", type="primary"):
    with st.spinner("Predicting…"):
        try:
            resp = requests.post(
                f"{API_URL}/predict",
                json={"hour": hour, "dayofweek": dayofweek, "month": month},
                timeout=5,
            )
            if resp.status_code == 200:
                prediction = resp.json()["predicted_energy_kW"]
                st.success(f"Predicted Energy: **{prediction:.3f} kW**")

                if prediction < 1.5:
                    st.info("🔵 Low — good consumption level")
                elif prediction < 3.0:
                    st.success("🟢 Moderate — normal consumption level")
                else:
                    st.warning("🔴 High — above typical consumption")

                now = datetime.datetime.now().strftime("%H:%M:%S")
                st.session_state.history.append({
                    "time": now,
                    "hour": hour,
                    "dayofweek": dayofweek,
                    "month": month,
                    "predicted_kW": prediction,
                })
            else:
                st.error(f"API error: {resp.status_code}")
        except Exception as e:
            st.error(f"Could not reach API: {e}")

if st.session_state.history:
    st.subheader("Prediction history (this session)")
    history_df = pd.DataFrame(st.session_state.history)
    st.line_chart(history_df, x="time", y="predicted_kW", height=200)
    with st.expander("Raw history"):
        st.dataframe(history_df)


# ── 24-hour forecast ─────────────────────────────────────────────────────────
st.divider()
st.header("24-hour Forecast")
st.write("Rolling prediction for the next 24 hours starting from the selected time above.")

if st.button("Run 24h Forecast"):
    with st.spinner("Forecasting…"):
        try:
            resp = requests.post(
                f"{API_URL}/forecast",
                json={"hour": hour, "dayofweek": dayofweek, "month": month},
                params={"horizon": 24},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()["forecast"]
                forecast_df = pd.DataFrame(data)
                forecast_df["label"] = forecast_df.apply(
                    lambda r: f"{['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][r['dayofweek']]} {r['hour']:02d}:00", axis=1
                )
                st.line_chart(forecast_df, x="label", y="predicted_energy_kW", height=280)
                with st.expander("Forecast data"):
                    st.dataframe(forecast_df[["label", "predicted_energy_kW"]])
            else:
                st.error(f"API error: {resp.status_code}")
        except Exception as e:
            st.error(f"Could not reach API: {e}")


# ── Historical actual consumption ────────────────────────────────────────────
st.divider()
st.header("Recent Actual Energy Consumption (last 3 days of data)")
try:
    df = pd.read_csv("data/processed/energy_clean.csv", parse_dates=["datetime"], index_col="datetime")
    cutoff = df.index.max() - pd.Timedelta(days=3)
    recent_df = df.loc[df.index >= cutoff]
    st.line_chart(recent_df["Global_active_power"], height=220)
    st.caption(f"Data range: {df.index.min().date()} → {df.index.max().date()} | {len(df):,} hourly records")
except Exception as e:
    st.error(f"Error loading data: {e}")
