import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from model_classes import SimplifiedGCN

model_path = "model_360_40"
num_node_features = 16
num_classes = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the CSV to get 'n' values for each run, assuming headerless CSV
labels_info_path = "data/labels_info_prediction.csv"
labels_info_df = pd.read_csv(labels_info_path, header=None, names=["run_name"])

model = SimplifiedGCN(num_node_features, num_classes)
model.load_state_dict(torch.load(f"data/1_pytorch_model/{model_path}.pth"))
model.eval()
model.to(device)

object_start = 1
object_end = 9

for object_num in range(object_start, object_end + 1):
    object_name = f"run_{object_num}"

    n_results = int(labels_info_df.iloc[object_num - 1]["run_name"])

    new_data = torch.load(f"data/torch_data_object_prediction/{object_name}.pt")
    if isinstance(new_data, Data):
        new_data = [new_data]
    data_loader = DataLoader(new_data, batch_size=1, shuffle=False)

    predictions = []
    for data in data_loader:
        data = data.to(device)
        with torch.no_grad():
            output = model(data.x, data.edge_index).squeeze()
            output = output.cpu().numpy().flatten()
            predictions.extend(output.tolist())

    # Slice the predictions to keep only the first 'n' results
    predictions = predictions[:n_results]

    # Save the sliced predictions
    prediction_file_path = (
        f"data/2_prediction_testing/{model_path}_predictions_{object_name}.csv"
    )
    np.savetxt(prediction_file_path, np.array(predictions), delimiter=",")
    print(f"Saved predictions for {model_path} to: {prediction_file_path}")
