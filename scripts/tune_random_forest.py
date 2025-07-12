import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import csv

# Load dataset
df = pd.read_csv("data/processed/featured_energy_weather_gb.csv")
print(f"📥 Initial dataset shape: {df.shape}")

# Drop column if it's entirely NaN
if "precipitation_probability" in df.columns and df["precipitation_probability"].isna().all():
    df = df.drop(columns=["precipitation_probability"])
    print("⚠️ Dropped 'precipitation_probability' (all values were NaN)")

# Drop rows with any NaNs
df = df.dropna()
print(f"✅ Cleaned dataset shape: {df.shape}")

# Check if data is non-empty
if df.empty:
    raise ValueError("❌ Dataset is empty after dropping NaNs!")

# Feature-target split
X = df.drop(columns=["time", "energy_load_mw"])
y = df["energy_load_mw"]

# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Model and hyperparameter grid
model = RandomForestRegressor(random_state=42)
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}

# GridSearchCV
print("🚀 Starting hyperparameter tuning...")
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv,
                    scoring="neg_mean_squared_error", verbose=2, n_jobs=-1)
grid.fit(X, y)

# Get best model
best_model = grid.best_estimator_

# Save best model
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/tuned_random_forest_model.pkl")
print("✅ Best model saved to models/tuned_random_forest_model.pkl")

# Evaluate best model
y_pred = best_model.predict(X)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print("\n🏆 Tuned Random Forest Performance:")
print(f"• MSE: {mse:.2f}")
print(f"• RMSE: {rmse:.2f}")
print(f"• R²: {r2:.4f}")

# Log results to CSV
os.makedirs("results", exist_ok=True)
csv_path = "results/performance_metrics.csv"

write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
with open(csv_path, mode="a", newline="") as file:
    writer = csv.writer(file)
    if write_header:
        writer.writerow(["Model", "MSE", "RMSE", "R2"])
    writer.writerow(["Tuned Random Forest", mse, rmse, r2])
