import streamlit as st
import requests
import datetime

st.set_page_config(page_title="Energy Prediction", page_icon=":sun_with_face:", layout="centered")

st.title("Home Energy ⚡ Consumption Forecast Dashboard🔋")
st.subheader("rendered by Taey👩‍💻")
st.write("This dashboard shows the predicted energy consumption for the next hour based on the weather and time of day!")

# get user input
col1, col2, col3 = st.columns(3)
with col1:
    hour = st.number_input("Hour (0-23)", min_value=0, max_value=23, value=0)
with col2:
    dayofweek = st.selectbox("Day of Week", options=list(range(7)), format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x])
with col3:
    month = st.selectbox("Month (1-12)", options=list(range(1, 13)), format_func=lambda x: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][x-1])

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
                st.success(f"Predicted Energy: {prediction:.3f} kW")
                st.bar_chart([prediction])
            else:
                st.error("Failed to make prediction. API error.")
        
        except Exception as e:
            st.error(f"An error occurred: {e}")