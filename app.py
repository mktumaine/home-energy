
# Save this as 'app.py' and run: streamlit run app.py

import streamlit as st
import joblib
import numpy as np

st.title("Smart Home Energy Consumption Prediction")

# Load saved model and encoders
model = joblib.load("le_season.joblib")
scaler = joblib.load("scaler.joblib")
le_appliance = joblib.load("le_appliance.pkl")
le_season = joblib.load("le_season.joblib")

# User inputs
appliance = st.selectbox("Appliance Type", le_appliance.classes_)
season = st.selectbox("Season", le_season.classes_)
temperature = st.number_input("Outdoor Temperature (°C)", -10, 50, 20)
household_size = st.slider("Household Size", 1, 10, 3)
hour = st.slider("Hour of Day", 0, 23, 12)
month = st.slider("Month", 1, 12, 6)
day = st.slider("Day", 1, 31, 15)

# Transform inputs
appliance_enc = le_appliance.transform([appliance])[0]
season_enc = le_season.transform([season])[0]

features = np.array([[appliance_enc, temperature, season_enc, household_size, 2023, month, day, hour]])
features_scaled = scaler.transform(features)

if st.button("Predict Energy Consumption"):
    prediction = model.predict(features_scaled)[0]
    st.success(f"Predicted Energy Consumption: {prediction:.2f} kWh")
