from utils import load_config
import torch
from data_loader import create_data_loader
from custom_dataset import CustomDataset
# Load settings from the config file
config = load_config()

# Extract settings from the config
data_dir = config['data_dir']
num_channels = config["num_channels"]
batch_size = config["batch_size"]

# Create data loaders
train_dataloader = create_data_loader("Train", batch_size)
test_dataloader = create_data_loader("Test", batch_size)

# Example usage:
if __name__ == "__main__":
    print("Loaded custom dataset.")
    print(f"Number of batches in the train DataLoader: {len(train_dataloader)}")
    print(f"Number of batches in the test DataLoader: {len(test_dataloader)}")

    for batch_idx, (features, labels) in enumerate(train_dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f"Features shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")
        break  # Stop after the first batch for demonstration