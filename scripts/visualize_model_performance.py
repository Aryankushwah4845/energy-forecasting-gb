import pandas as pd
import matplotlib.pyplot as plt
import os

# Load metrics
df = pd.read_csv("results/performance_metrics.csv")

# Create output folder
os.makedirs("visuals", exist_ok=True)

# RMSE Plot
plt.figure(figsize=(8, 5))
plt.bar(df["Model"], df["RMSE"], color="skyblue")
plt.title("Model Comparison – RMSE")
plt.ylabel("Root Mean Squared Error")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("visuals/rmse_comparison.png")
plt.close()

# R² Plot
plt.figure(figsize=(8, 5))
plt.bar(df["Model"], df["R2"], color="lightgreen")
plt.title("Model Comparison – R² Score")
plt.ylabel("R² Score")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("visuals/r2_comparison.png")
plt.close()

print("✅ Visualizations saved to 'visuals/' folder.")
