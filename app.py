# app.py
import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Smart Home Energy Predictor", layout="centered")
st.title("Smart Home Energy Consumption Prediction")

# -----------------------------
# Load model, scaler, and encoders
# -----------------------------
try:
    model = joblib.load("energy_model.joblib")
    scaler = joblib.load("scaler.joblib")
    le_appliance = joblib.load("le_appliance.joblib")
    le_season = joblib.load("le_season.joblib")
except Exception as e:
    st.error(f"Error loading model or encoders: {e}")
    st.stop()

# -----------------------------
# User inputs
# -----------------------------
appliance = st.selectbox("Appliance Type", le_appliance.classes_)
season = st.selectbox("Season", le_season.classes_)
temperature = st.number_input("Outdoor Temperature (°C)", -10, 50, 20)
household_size = st.slider("Household Size", 1, 10, 3)
hour = st.slider("Hour of Day", 0, 23, 12)
month = st.slider("Month", 1, 12, 6)
day = st.slider("Day", 1, 31, 15)
year = st.number_input("Year", 2023, 2100, 2023)  # include Year if used in training

# -----------------------------
# Encode categorical variables
# -----------------------------
appliance_enc = le_appliance.transform([appliance])[0]
season_enc = le_season.transform([season])[0]

# -----------------------------
# Create feature array
# -----------------------------
# Make sure this order exactly matches the training columns in retrain_model.py:
# ['Appliance Type', 'Season', 'Temperature', 'Household Size', 'Year', 'Month', 'Day', 'Hour']
features = [appliance_enc, season_enc, temperature, household_size, year, month, day, hour]
features_array = np.array(features).reshape(1, -1)

# -----------------------------
# Scale features
# -----------------------------
try:
    features_scaled = scaler.transform(features_array)
except ValueError as e:
    st.error(f"Feature scaling error: {e}")
    st.stop()

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict Energy Consumption"):
    try:
        prediction = model.predict(features_scaled)[0]
        st.success(f"Predicted Energy Consumption: {prediction:.2f} kWh")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

    # Optional: show input summary
    st.subheader("Input Summary")
    st.write({
        "Appliance": appliance,
        "Season": season,
        "Temperature (°C)": temperature,
        "Household Size": household_size,
        "Year": year,
        "Month": month,
        "Day": day,
        "Hour": hour
    })
