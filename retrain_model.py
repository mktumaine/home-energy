# retrain_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# 1️⃣ Load Dataset
# -----------------------------
df = pd.read_csv("smart home energy consumption large.csv")

# -----------------------------
# 2️⃣ Preprocess Data
# -----------------------------
# Convert Date and Time
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M").dt.hour
df.drop(columns=["Date","Time"], inplace=True)

# Encode categorical variables
le_appliance = LabelEncoder()
df["Appliance Type"] = le_appliance.fit_transform(df["Appliance Type"])
joblib.dump(le_appliance, "le_appliance.joblib")

le_season = LabelEncoder()
df["Season"] = le_season.fit_transform(df["Season"])
joblib.dump(le_season, "le_season.joblib")

# Features and target
X = df.drop(columns=["Energy Consumption (kWh)", "Home ID"])
y = df["Energy Consumption (kWh)"]

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.joblib")

# -----------------------------
# 3️⃣ Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4️⃣ Train Smaller Random Forest
# -----------------------------
model = RandomForestRegressor(
    n_estimators=50,  # smaller model for GitHub/Streamlit
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# -----------------------------
# 5️⃣ Evaluate Model
# -----------------------------
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Model Evaluation:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# -----------------------------
# 6️⃣ Save Compressed Model
# -----------------------------
joblib.dump(model, "energy_model.joblib", compress=9)
print("Compressed model saved as energy_model.joblib (~25 MB)")