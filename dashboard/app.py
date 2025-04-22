import streamlit as st
import requests
import datetime
import pandas as pd
st.set_page_config(page_title="Energy Prediction", page_icon="🔋", layout="centered")

st.title("⚡  Home Energy Consumption Forecast Dashboard")
st.subheader("by Taey👩‍💻")
st.write("This dashboard shows the predicted energy consumption for the next hour based on the weather and time of day!")

# get user input
col1, col2, col3 = st.columns(3)
with col1:
    hour = st.number_input("Hour (0-23)", min_value=0, max_value=23, value=0)
with col2:
    dayofweek = st.selectbox("Day of Week", options=list(range(7)), format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x])
with col3:
    month = st.selectbox("Month (1-12)", options=list(range(1, 13)), format_func=lambda x: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][x-1])

# session state
if "history" not in st.session_state:
    st.session_state.history = []

# make prediction
if st.button("Predict"):
    with st.spinner("Predicting..."):
        try:
            response = requests.post(
                "http://localhost:8000/predict",
                json={"hour": hour, "dayofweek": dayofweek, "month": month}
            )
            if response.status_code == 200:
                prediction = response.json()["predicted_energy_in_kW"]
                st.success(f"Predicted Energy: **{prediction:.3f} kW**")
                
                # state message
                if prediction < 1.5:
                    st.info("🔵 Low energy consumption - Good Consumption")
                elif prediction < 3.0:
                    st.success("🟢 Normal energy consumption - Moderate Consumption")
                else:
                    st.error("🔴 High energy consumption - High Consumption")
                
                # save and update history
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.history.append({"timestamp": now, "value": prediction})
                
                st.bar_chart([prediction])
            
            else:
                st.error("Failed to make prediction. API error.")
        
        except Exception as e:
            st.error(f"An error occurred: {e}")

# visualize prediction history
if st.session_state.history:
    st.subheader("📊 Prediction History")
    history_df = pd.DataFrame(st.session_state.history)
    st.line_chart(history_df, x="timestamp", y="value", height=200)

# visualize recent predictions
st.subheader("📈 Recent Actual Energy Consumption")
try:
    df = pd.read_csv("data/processed/energy_clean.csv")
    df.index = pd.to_datetime(df.index)
    recent_df = df.loc[df.index >= (df.index.max() - pd.Timedelta(days=3))]
    st.line_chart(recent_df["Global_active_power"])
except Exception as e:
    st.error(f"Error loading data: {e}")

# # visualize weather data
# st.subheader("🌤️ Weather Data")
# try:
    