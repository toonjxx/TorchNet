import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from utils import load_config


# Define a custom PyTorch dataset class for your time series data
class CustomDataset(Dataset):
    def __init__(self, csv_files, labels_file):
        self.csv_files = csv_files
        self.labels_df = pd.read_csv(labels_file)

        # Load settings from the config file
        config = load_config()
        self.data_dir = config["data_dir"]

    def __len__(self):
        return len(self.csv_files)

    def __getitem__(self, idx):
        csv_file = self.csv_files[idx]
        df = self.load_sample_csv(csv_file)

        # Extract features (time series data) and labels
        features = df.iloc[:, :].values  # Assuming the last 4 columns are labels

        # Extract labels based on the sample file's name (assuming a common format)
        sample_id = os.path.basename(csv_file).split(".")[0]
        labels_row = self.labels_df[self.labels_df["ID"] == sample_id]

        if not labels_row.empty:
            labels = labels_row.iloc[:, 1:].values
        else:
            labels = None  # Handle cases where labels are not found

        # Convert to PyTorch tensors
        features = torch.FloatTensor(features)
        if labels is not None:
            labels = torch.FloatTensor(labels)

        return features, labels

    def load_sample_csv(self, sample_csv_file):
        try:
            df = pd.read_csv(sample_csv_file)

            return df
        except Exception as e:
            print(f"Error loading CSV file {sample_csv_file}: {str(e)}")
            return None