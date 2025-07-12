import pandas as pd
import os

# Paths
input_path = "data/processed/clean_energy_weather_gb.csv"
output_path = "data/processed/featured_energy_weather_gb.csv"

# Load dataset
df = pd.read_csv(input_path, parse_dates=["time"])
df = df.sort_values("time")

# Create lag features
df["load_lag_1h"] = df["energy_load_mw"].shift(1)
df["load_lag_3h"] = df["energy_load_mw"].shift(3)

# Create rolling mean and std features
df["load_roll_mean_3h"] = df["energy_load_mw"].rolling(window=3).mean()
df["load_roll_std_3h"] = df["energy_load_mw"].rolling(window=3).std()
df["load_roll_mean_6h"] = df["energy_load_mw"].rolling(window=6).mean()
df["load_roll_std_6h"] = df["energy_load_mw"].rolling(window=6).std()

# Check before dropping
before_drop = len(df)
df = df.dropna(subset=[
    "load_lag_1h", "load_lag_3h",
    "load_roll_mean_3h", "load_roll_std_3h",
    "load_roll_mean_6h", "load_roll_std_6h"
])
after_drop = len(df)

print(f"ℹ️ Dropped {before_drop - after_drop} rows due to missing values from lag/rolling features.")

# Save result
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)
print(f"✅ Feature-engineered data saved to {output_path}")
