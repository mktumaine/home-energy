
# Save this as 'app.py' and run: streamlit run app.py

import streamlit as st
import joblib
import numpy as np

st.title("Smart Home Energy Consumption Prediction")

# Load saved model and encoders
model = joblib.load("le_season.joblib")
scaler = joblib.load("scaler.joblib")
le_appliance = joblib.load("le_appliance.joblib")
le_season = joblib.load("le_season.joblib")

# User inputs
Appliance = st.selectbox("Appliance Type", le_appliance.classes_)
Season = st.selectbox("Season", le_season.classes_)
Temperature = st.number_input("Outdoor Temperature (°C)", -10, 50, 20)
Household_size = st.slider("Household Size", 1, 10, 3)
Hour = st.slider("Hour of Day", 0, 23, 12)
Month = st.slider("Month", 1, 12, 6)
Day = st.slider("Day", 1, 31, 15)

# Transform inputs
appliance_enc = le_appliance.transform([appliance])[0]
season_enc = le_season.transform([season])[0]

features = np.array([[Appliance_enc, Temperature, Season_enc, Household_size, 2023, Month, Day, Hour of Day]])
features_scaled = scaler.transform(features)

if st.button("Predict Energy Consumption"):
    prediction = model.predict(features_scaled)[0]
    st.success(f"Predicted Energy Consumption: {prediction:.2f} kWh")
