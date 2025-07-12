import requests
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Set parameters
country_code = "GB"
start_date = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")
end_date = datetime.utcnow().strftime("%Y-%m-%d")

# Weather API endpoint
url = "https://archive-api.open-meteo.com/v1/archive"

# API parameters for 6 months of hourly weather data
params = {
    "latitude": 51.5,  # London
    "longitude": -0.1,
    "start_date": start_date,
    "end_date": end_date,
    "hourly": "temperature_2m,cloudcover,windspeed_10m,precipitation_probability",
    "timezone": "Europe/London"
}

# Fetch weather data
print("🔄 Fetching weather data...")
response = requests.get(url, params=params)
data = response.json()

# Convert to DataFrame
weather_df = pd.DataFrame(data["hourly"])
weather_df["time"] = pd.to_datetime(weather_df["time"])

# Save weather data
os.makedirs("data/raw", exist_ok=True)
weather_df.to_json("data/raw/weather_gb.json", orient="records", lines=False)
print("✅ Weather data saved to data/raw/weather_gb.json")

# Generate mock energy data for the same time range
print("⚡ Generating mock energy load data...")
energy_df = pd.DataFrame({
    "time": weather_df["time"],
    "energy_load_mw": np.random.normal(loc=42000, scale=3500, size=len(weather_df)).round(2)
})

# Save energy data
energy_df.to_json("data/raw/energy_mock_gb.json", orient="records", lines=False)
print("✅ Energy mock data saved to data/raw/energy_mock_gb.json")
