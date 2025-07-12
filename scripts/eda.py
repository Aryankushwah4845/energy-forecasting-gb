import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load your cleaned data
df = pd.read_csv("data/processed/clean_energy_weather_gb.csv", parse_dates=["time"])

# Create output folder
os.makedirs("eda_outputs", exist_ok=True)

# Plot 1: Energy Load over Time
plt.figure(figsize=(14, 5))
plt.plot(df["time"], df["energy_load_mw"], color="blue")
plt.title("Energy Load Over Time")
plt.xlabel("Time")
plt.ylabel("Load (MW)")
plt.tight_layout()
plt.savefig("eda_outputs/energy_load_over_time.png")
plt.close()

# Plot 2: Temperature vs Energy Load
plt.figure(figsize=(7, 5))
sns.scatterplot(x="temperature_2m", y="energy_load_mw", data=df, alpha=0.3)
plt.title("Temperature vs. Energy Load")
plt.tight_layout()
plt.savefig("eda_outputs/temperature_vs_energy.png")
plt.close()

# Plot 3: Weekday Load
plt.figure(figsize=(8, 5))
sns.boxplot(x="weekday", y="energy_load_mw", data=df)
plt.title("Energy Load by Weekday")
plt.tight_layout()
plt.savefig("eda_outputs/load_by_weekday.png")
plt.close()

# Plot 4: Correlation Heatmap
plt.figure(figsize=(8, 6))
corr = df[["temperature_2m", "cloudcover", "windspeed_10m", "precipitation_probability", "energy_load_mw"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("eda_outputs/correlation_heatmap.png")
plt.close()

print("✅ EDA plots saved in the 'eda_outputs' folder.")
