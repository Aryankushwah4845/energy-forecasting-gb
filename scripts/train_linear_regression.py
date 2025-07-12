import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import numpy as np

# Load the feature-engineered dataset
df = pd.read_csv("data/processed/featured_energy_weather_gb.csv")
print(f"🔍 Raw Data shape: {df.shape}")

# Drop irrelevant or problematic columns
if "precipitation_probability" in df.columns and df["precipitation_probability"].isna().all():
    df = df.drop(columns=["precipitation_probability"])
    print("⚠️ Dropped 'precipitation_probability' column (all values were NaN).")

# Drop rows with any NaN values
df = df.dropna()
print(f"✅ Cleaned Data shape: {df.shape} (dropped rows with NaNs)")

# Define features and target
X = df.drop(columns=["time", "energy_load_mw"])
y = df["energy_load_mw"]
print(f"✅ Feature matrix shape: {X.shape}")
print(f"✅ Target vector shape: {y.shape}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Linear Regression Results:")
print(f"• Mean Squared Error: {mse:.2f}")
print(f"• R² Score: {r2:.4f}")

# Save the model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/linear_regression_model.pkl")
print("✅ Model saved to models/linear_regression_model.pkl")

import csv
import os
os.makedirs("results", exist_ok=True)

with open("results/performance_metrics.csv", mode="a", newline="") as file:
    writer = csv.writer(file)
    if file.tell() == 0:  # if file is empty
        writer.writerow(["Model", "MSE", "RMSE", "R2"])
    writer.writerow(["Linear Regression", mse, np.sqrt(mse), r2])


