# scripts/compare_models.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("results/performance_metrics.csv")

# Plot RMSE Comparison
plt.figure(figsize=(8, 5))
plt.bar(df["Model"], df["RMSE"], color="skyblue")
plt.title("Model Comparison - RMSE")
plt.ylabel("Root Mean Squared Error")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/rmse_comparison.png")

# Plot R² Comparison
plt.figure(figsize=(8, 5))
plt.bar(df["Model"], df["R2"], color="green")
plt.title("Model Comparison - R² Score")
plt.ylabel("R² Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/r2_comparison.png")

print("✅ Comparison plots saved to 'results/'")
