import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load data
df = pd.read_csv("data/processed/featured_energy_weather_gb.csv")
print(f"🔍 Raw Data shape: {df.shape}")

# Drop unused or problematic columns
if 'precipitation_probability' in df.columns:
    if df['precipitation_probability'].isna().all():
        df.drop(columns=['precipitation_probability'], inplace=True)
        print("⚠️ Dropped 'precipitation_probability' column (all values were NaN).")

# Drop rows with NaNs (from lag/rolling features)
df = df.dropna()
print(f"✅ Cleaned Data shape: {df.shape}")

# Prepare features and target
X = df.drop(columns=["time", "energy_load_mw"])
y = df["energy_load_mw"]

print(f"✅ Feature matrix shape: {X.shape}")
print(f"✅ Target vector shape: {y.shape}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Random Forest Regression Results:")
print(f"• Mean Squared Error: {mse:.2f}")
print(f"• R² Score: {r2:.4f}")

# Save model
joblib.dump(model, "models/random_forest_model.pkl")
print("✅ Model saved to models/random_forest_model.pkl")

import csv
import os
os.makedirs("results", exist_ok=True)

with open("results/performance_metrics.csv", mode="a", newline="") as file:
    writer = csv.writer(file)
    if file.tell() == 0:
        writer.writerow(["Model", "MSE", "RMSE", "R2"])
    writer.writerow(["Random Forest", mse, np.sqrt(mse), r2]) 


