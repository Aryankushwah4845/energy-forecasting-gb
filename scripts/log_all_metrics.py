import csv
import numpy as np

# Final metrics (replace these with your actual values)
metrics = [
    ["Linear Regression", 6210673.00, np.sqrt(6210673.00), 0.5091],
    ["Random Forest", 7319565.35, np.sqrt(7319565.35), 0.4215],
    ["XGBoost", 7076684.63, np.sqrt(7076684.63), 0.4407],
    ["Tuned Random Forest", 5890324.19, np.sqrt(5890324.19), 0.5096]
]

# Write to CSV
with open("results/performance_metrics.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Model", "MSE", "RMSE", "R2"])
    writer.writerows(metrics)

print("✅ Final performance_metrics.csv file created.")
