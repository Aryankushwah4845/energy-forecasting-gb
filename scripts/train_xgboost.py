import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Load the data
df = pd.read_csv("data/processed/featured_energy_weather_gb.csv")
print(f"🔍 Raw Data shape: {df.shape}")

# Drop columns with all NaNs (like 'precipitation_probability')
df = df.dropna(axis=1, how='all')
if 'precipitation_probability' not in df.columns:
    print("⚠️ Dropped 'precipitation_probability' column (all values were NaN).")

# Drop rows with any NaNs (from rolling or lagged features)
df_clean = df.dropna()
print(f"✅ Cleaned Data shape: {df_clean.shape} (dropped {df.shape[0] - df_clean.shape[0]} rows with NaNs)")

# Feature selection
X = df_clean.drop(columns=["time", "energy_load_mw"])
y = df_clean["energy_load_mw"]

print(f"✅ Feature matrix shape: {X.shape}")
print(f"✅ Target vector shape: {y.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost Regressor
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 XGBoost Regression Results:")
print(f"• Mean Squared Error: {mse:.2f}")
print(f"• R² Score: {r2:.4f}")

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgboost_model.pkl")
print("✅ Model saved to models/xgboost_model.pkl")

import csv
import os
os.makedirs("results", exist_ok=True)

with open("results/performance_metrics.csv", mode="a", newline="") as file:
    writer = csv.writer(file)
    if file.tell() == 0:
        writer.writerow(["Model", "MSE", "RMSE", "R2"])
    writer.writerow(["XGBoost", mse, np.sqrt(mse), r2]) 



