import os
from torch.utils.data import DataLoader, Dataset
from utils import load_config
from pathlib import Path
import torch
from sklearn.model_selection import KFold
import pandas as pd



def create_data_loader(split, batch_size):
    config = load_config()
    data_dir = config["data_dir"]
    data_dir = os.path.abspath(data_dir)

    data_dir = os.path.join(data_dir,split)
    labels_file = Path(os.path.join(data_dir,'Label.csv'))
    dataset = Build_Dataset(data_dir, labels_file,label_select="Hypertension")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,pin_memory=True)
    return dataloader

def train_val_dataloader_kfold(split, batch_size, k=5):
    config = load_config()
    data_dir = config["data_dir"]
    data_dir = os.path.abspath(data_dir)
    data_dir = os.path.join(data_dir,split)
    labels_file = Path(os.path.join(data_dir,'Label.csv'))
    dataset = Build_Dataset(data_dir, labels_file,label_select="Hypertension")
    kf = KFold(n_splits=k, shuffle=True)
    fold_indices = list(kf.split(dataset)) # type: ignore

    # Create k pairs of dataloaders
    dataloaders = []
    for i in range(k):
        train_indices, val_indices = fold_indices[i]
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,pin_memory=True)
        dataloaders.append((train_dataloader, val_dataloader))
    return dataloaders

class Build_Dataset(Dataset):
    def __init__(self, data_dir, labels_file,label_select,transform=None):
        """
        Arguments:
            labels_file (string): Path to the csv file with labels.
            data_dir (string): Directory with all the  sample.
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
                SBP = (labels_row["SBP"]-30)/170
                DBP = (labels_row["DBP"]-30)/170
                labels = pd.array([SBP,DBP])
            elif label_select == "Hypertension":
                isHyper = labels_row["Hypertension"]
                labels = pd.array([isHyper])
            elif label_select == "SBP":
                SBP = (labels_row["SBP"]-30)/170
                labels = pd.array([SBP])
            elif label_select == "DBP":
                DBP = (labels_row["DBP"]-30)/170
                labels = pd.array([DBP])
            else:
                labels = None
        else:
            labels = None  # Handle cases where labels are not found

        # Convert to PyTorch tensors
        if data is not None:
            data = torch.FloatTensor(data.values)
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