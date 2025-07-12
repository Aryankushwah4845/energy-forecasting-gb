import pandas as pd
import json
import os

# Load weather data
with open("data/raw/weather_gb.json", "r") as f:
    weather_data = json.load(f)

# Convert list of dicts to DataFrame directly
weather_df = pd.DataFrame(weather_data)

# Load energy data
with open("data/raw/energy_mock_gb.json", "r") as f:
    energy_data = json.load(f)

energy_df = pd.DataFrame(energy_data)

# Convert time columns to datetime
weather_df["time"] = pd.to_datetime(weather_df["time"])
energy_df["time"] = pd.to_datetime(energy_df["time"])

# Merge on time
merged_df = pd.merge(weather_df, energy_df, on="time", how="inner")

# Extract temporal features
merged_df["hour"] = merged_df["time"].dt.hour
merged_df["day"] = merged_df["time"].dt.day
merged_df["weekday"] = merged_df["time"].dt.weekday
merged_df["month"] = merged_df["time"].dt.month

# Save merged cleaned file
os.makedirs("data/processed", exist_ok=True)
output_path = "data/processed/clean_energy_weather_gb.csv"
merged_df.to_csv(output_path, index=False)

print(f"✅ Cleaned and merged data saved to {output_path}")
