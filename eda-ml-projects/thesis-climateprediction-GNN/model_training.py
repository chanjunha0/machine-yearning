import os
import torch
import pandas as pd
from torch_geometric.data import Data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import csv
from model_classes import (
    SimplifiedGCN,
)
import torch.optim as optim
import datetime
from typing import Any

## Constants

# Path for Pytorch data objects
folder_path = "data/torch_data_object_training"

# Path to the CSV containing labeled nodes info
labels_info_path = "data/torch_data_object_training/labels_info.csv"

model_path = "data/1_pytorch_model/model.pth"

# Path to save losses
losses_file_path = "data/training_and_test_losses.csv"

# Model type
model_type = SimplifiedGCN

# Placeholder value for unlabeled nodes
placeholder_value = -1


## Functions


def load_graph_data_objects(
    folder_path: str, ranges: list[tuple[int, int]]
) -> list[list[Data]]:
    """
    Load graph data objects from a folder based on specified ranges.

    Args:
        folder_path (str): Path to the folder containing graph data objects.
        ranges (list of tuple): List of tuples where each tuple contains the start and end run numbers.

    Returns:
        list: A list of lists, where each inner list contains graph data objects for a specific town.
    """
    # List to hold groups of simulations by town
    graph_data_objects_by_town = []

    for start_run, end_run in ranges:
        # List to hold simulations for the current town
        town_simulations = []

        # Load each simulation for the current town based on the specified range
        for run_number in range(start_run, end_run + 1):
            expected_file = f"run_{run_number}.pt"
            graph_path = os.path.join(folder_path, expected_file)
            if os.path.exists(graph_path):
                graph_data = torch.load(graph_path)
                if isinstance(graph_data, Data):
                    town_simulations.append(graph_data)
            else:
                print(f"File does not exist: {graph_path}")

        # Add the current town's simulations to the main list
        graph_data_objects_by_town.append(town_simulations)

    return graph_data_objects_by_town


def load_or_initialize_model(
    model_path: str, model_class: type, *model_args: Any, **model_kwargs: Any
) -> Any:
    """
    Load a previously trained model or initialize a new one.

    Args:
        model_path (str): Path to the model file.
        model_class (class): The model class to be instantiated.
        *model_args: Variable length argument list for the model class.
        **model_kwargs: Arbitrary keyword arguments for the model class.

    Returns:
        object: An instance of the model class, either loaded with pre-trained weights or newly initialized.
    """
    if os.path.exists(model_path):
        model = model_class(*model_args, **model_kwargs)
        model.load_state_dict(torch.load(model_path))
        print("Loaded previously trained model.")
    else:
        model = model_class(*model_args, **model_kwargs)
        print("Initialized new model.")
    return model


def load_num_labeled_nodes(labels_info_path: str) -> list:
    """
    Load the number of labeled nodes from a CSV file.

    Args:
        labels_info_path (str): Path to the CSV file containing label information.

    Returns:
        list: A list of labeled nodes.
    """
    labels_info = pd.read_csv(labels_info_path, usecols=[0])
    return labels_info.iloc[:, 0].tolist()


def count_graph_data_objects(folder_path: str) -> int:
    """
    Count the number of graph data objects in a folder.

    Args:
        folder_path (str): Path to the folder containing graph data objects.

    Returns:
        int: The number of graph data objects (files with .pt extension) in the folder.
    """
    count = 0
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pt"):
            count += 1
    return count


def train(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, batch: Data
) -> float:
    """
    Train the model on a given batch of data.

    Args:
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        batch (torch_geometric.data.Data): The batch of data for training.

    Returns:
        float: The training loss value.
    """
    model.train()
    optimizer.zero_grad()

    out = model(batch.x, batch.edge_index)
    out = out.squeeze()

    sensor_out = out[: batch.train_mask.sum().item()]

    valid_out = sensor_out
    valid_labels = batch.y[batch.train_mask]

    loss = F.mse_loss(valid_out, valid_labels)
    loss.backward()
    optimizer.step()

    return loss.item()


def test(model: torch.nn.Module, batch: Data) -> float:
    """
    Evaluate the model on a given batch of data.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        batch (torch_geometric.data.Data): The batch of data for testing.

    Returns:
        float: The test loss value.
    """
    model.eval()
    with torch.no_grad():
        out = model(batch.x, batch.edge_index)
        out = out.squeeze()

        labeled_out = out[: len(batch.y)]

        test_out = labeled_out[batch.test_mask]
        valid_labels = batch.y[batch.test_mask]

        test_loss = F.mse_loss(test_out, valid_labels).item()

    return test_loss


# Record Start Time
start_time = time.time()
print(start_time)

total_objects = 360

# Generate ranges to load each object one by one
ranges = [(i, i) for i in range(1, total_objects + 1)]

# Load graph data objects
graph_data_objects = load_graph_data_objects(folder_path, ranges)

# Load the number of labeled nodes for each graph
num_labeled_nodes_list = load_num_labeled_nodes(labels_info_path)

# Model Settings
if graph_data_objects and graph_data_objects[0]:
    num_node_features = graph_data_objects[0][0].num_node_features
    print(f"Number of node features: {num_node_features}")
else:
    print("Error: No graph data objects loaded.")

num_classes = 1

criterion = torch.nn.MSELoss()

# CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_or_initialize_model(
    model_path, model_type, num_node_features, num_classes
).to(device)
print(f"Model {model_type.__name__} is ready for training.")
if device.type == "cuda":
    print(f"Model is using GPU: {torch.cuda.get_device_name(device)}")
else:
    print("Model is using CPU")


# Set optimizer and learning rate
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)


extended_test_mask = None


## Training and Testing Loop

train_losses, test_losses, average_epoch_losses = [], [], []

# Record Start Time before the loop
total_start_time = time.time()

for epoch in range(50):
    epoch_start_time = time.time()

    epoch_loss = 0
    for (
        data_list
    ) in (
        graph_data_objects
    ):  # 'graph_data_objects' is a list of lists of 'Data' objects
        for data in data_list:
            data = data.to(device)
            train_loss = train(model, optimizer, data)
            epoch_loss += train_loss
    average_loss = epoch_loss / sum(len(data_list) for data_list in graph_data_objects)
    average_epoch_losses.append(average_loss)

    # Calculate test loss at the end of the epoch
    model.eval()
    test_epoch_loss = 0
    with torch.no_grad():
        for data_list in graph_data_objects:
            for data in data_list:
                data = data.to(device)
                test_loss = test(model, data)
                test_epoch_loss += test_loss
    average_test_loss = test_epoch_loss / sum(
        len(data_list) for data_list in graph_data_objects
    )
    test_losses.append(average_test_loss)

    # Calculate and print runtime for the epoch
    epoch_runtime = time.time() - epoch_start_time

    # Calculate total time elapsed
    total_elapsed_time = time.time() - total_start_time

    print(
        f"Epoch {epoch+1}, Average Train Loss: {average_loss:.2f}, Average Test Loss: {average_test_loss:.2f}, Runtime: {epoch_runtime:.2f} seconds, Total Elapsed Time: {(total_elapsed_time)/60} minutes"
    )


# Plotting
plt.figure(figsize=(10, 6))

# Plotting the average training loss per epoch.
plt.plot(
    range(1, len(average_epoch_losses) + 1),
    average_epoch_losses,
    marker="o",
    linestyle="-",
    color="b",
    label="Training Loss",
)

# Plotting the average test loss per epoch.
plt.plot(
    range(1, len(test_losses) + 1),
    test_losses,
    marker="s",
    linestyle="--",
    color="r",
    label="Test Loss",
)


## Save the Model
file_path = rf"data\1_pytorch_model\model.pth"

# Save the model
torch.save(model.state_dict(), file_path)

# Record End Time
end_time = time.time()

# Calculate Training Duration
training_duration = end_time - start_time
minutes, seconds = divmod(training_duration, 60)
hours, minutes = divmod(minutes, 60)
print(
    f"Training took {int(hours)} hours, {int(minutes)} minutes, and {seconds} seconds."
)

# Get the current date and time
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Update the CSV file name to include the timestamp
losses_file_path = f"data/training_and_test_losses_{timestamp}.csv"

# Save the train_losses and test_losses to the CSV file
with open(losses_file_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Training Loss", "Test Loss"])
    for epoch, (train_loss, test_loss) in enumerate(
        zip(average_epoch_losses, test_losses), start=1
    ):
        writer.writerow([epoch, train_loss, test_loss])

print("Losses saved.")

plt.title("Average Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
