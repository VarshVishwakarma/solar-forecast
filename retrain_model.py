import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
df = pd.read_csv("data/final_solar_dataset.csv")

# Convert timestamp to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Create time features
df["hour"] = df["timestamp"].dt.hour
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# Create lag features
df["power_t_1"] = df["power_output"].shift(1)
df["power_t_2"] = df["power_output"].shift(2)

# Drop rows with NaN (from lag features)
df = df.dropna()

# Features and target
X = df[["temperature", "humidity", "ghi", "hour_sin", "hour_cos", "power_t_1", "power_t_2"]]
y = df["power_output"]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_scaled, y)

# Save new version
joblib.dump(model, "app/model_v2.joblib")
joblib.dump(scaler, "app/scaler_v2.joblib")

print("New model version v2.0 saved successfully")
