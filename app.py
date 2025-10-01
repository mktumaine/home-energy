import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Smart Home Energy Predictor", layout="centered")
st.title("Smart Home Energy Consumption Prediction")

# -----------------------------
# Load compressed model and encoders
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

# -----------------------------
# Encode categorical variables
# -----------------------------
appliance_enc = le_appliance.transform([appliance])[0]
season_enc = le_season.transform([season])[0]

# -----------------------------
# Create feature array (order must match training)
# -----------------------------
features = [appliance_enc, season_enc, temperature, household_size, hour, month, day]
features_array = np.array(features).reshape(1, -1)

# -----------------------------
# Scale features
# -----------------------------
features_scaled = scaler.transform(features_array)

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
        "Hour of Day": hour,
        "Month": month,
        "Day": day
    })
