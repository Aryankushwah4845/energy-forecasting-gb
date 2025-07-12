import os
import csv
import numpy as np

def log_metrics(model_name, mse, r2):
    os.makedirs("results", exist_ok=True)
    file_path = "results/performance_metrics.csv"

    write_header = not os.path.exists(file_path)

    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["Model", "MSE", "RMSE", "R2"])
        writer.writerow([model_name, mse, np.sqrt(mse), r2])
