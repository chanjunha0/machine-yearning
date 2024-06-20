import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

# Load configuration from config.json
with open("config.json", "r") as config_file:
    config = json.load(config_file)

evaluation_config = config["evaluation"]
model_name = evaluation_config["model_name"]
start_run = evaluation_config["start_run"]
end_run = evaluation_config["end_run"]

runs = [f"run_{i}" for i in range(start_run, end_run + 1)]

# Initialize a DataFrame to store results
results_df = pd.DataFrame(columns=["Run", "MAE", "RMSE", "R2"])

for run_num in runs:
    # Actual simulated labels
    label_actual_df = pd.read_csv(
        f"data/2_prediction_testing/actual_label_{run_num}.csv", header=None
    )
    # Predicted labels
    label_predict_df = pd.read_csv(
        f"data/2_prediction_testing/{model_name}_predictions_{run_num}.csv", header=None
    )

    actual_values = label_actual_df.iloc[:, 0]
    predicted_values = label_predict_df.iloc[:, 0]

    # Calculate MAE
    mae = mean_absolute_error(actual_values, predicted_values)

    # Calculate RMSE
    rmse_value = np.sqrt(((predicted_values - actual_values) ** 2).mean())

    # Calculate R2
    r2 = r2_score(actual_values, predicted_values)

    # Append results to DataFrame
    new_row = pd.DataFrame(
        {"Run": [run_num], "MAE": [mae], "RMSE": [rmse_value], "R2": [r2]}
    )
    results_df = pd.concat([results_df, new_row], ignore_index=True)


# Calculate averages of each metric
avg_metrics = results_df.mean(numeric_only=True)

# Creating a new row with these averages
avg_row = pd.DataFrame(
    {
        "Run": ["Average"],
        "MAE": [avg_metrics["MAE"]],
        "RMSE": [avg_metrics["RMSE"]],
        "R2": [avg_metrics["R2"]],
    }
)

# Append the average row to the DataFrame
results_df = pd.concat([results_df, avg_row], ignore_index=True)

print(results_df)

# Save to CSV
results_df.to_csv(f"model_performance_metrics_{model_name}.csv", index=False)
