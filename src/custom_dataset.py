import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Define a custom PyTorch dataset class for your time series data
class CustomDataset(Dataset):
    def __init__(self, data_dir, labels_file,label_select,transform=None):
        """
        Arguments:
            labels_file (string): Path to the csv file with labels.
            data_dir (string): Directory with all the sample.
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        self.data_dir = data_dir
        self.labels_df = pd.read_csv(labels_file)
        self.label_select = label_select

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        sample_id = self.labels_df["IDSample"][idx]
        label_select = self.label_select
        data = self.load_data(os.path.join(self.data_dir,f"{sample_id}.csv"))

        labels_row = self.labels_df.iloc[idx,:]
        if not labels_row.empty:
            if label_select == "BP":
                SBP = labels_row["SBP"]
                DBP = labels_row["DBP"]
                labels = pd.array([SBP,DBP])

            elif label_select == "Hypertension":
                labels = labels_row["Hypertension"]
            else:
                labels = None
        else:
            labels = None  # Handle cases where labels are not found

        # Convert to PyTorch tensors
        data = torch.FloatTensor(data.values)
        data = data.permute(1,0)
        if labels is not None:
            labels = torch.FloatTensor(labels)
        return data, labels

    def load_data(self, sample_csv_file):
        try:
            df = pd.read_csv(sample_csv_file,header=None)
            return df
        except Exception as e:
            print(f"Error loading CSV file {sample_csv_file}: {str(e)}")
            return None
        
