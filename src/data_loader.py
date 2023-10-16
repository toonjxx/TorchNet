import os
from random import shuffle
from joblib import PrintTime
from torch.utils.data import DataLoader, random_split
from custom_dataset import CustomDataset
from utils import load_config
from pathlib import Path
import torch
from sklearn.model_selection import KFold

def create_data_loader(split, batch_size):
    config = load_config()
    data_dir = config["data_dir"]
    data_dir = os.path.abspath(data_dir)

    data_dir = os.path.join(data_dir,split)
    labels_file = Path(os.path.join(data_dir,'Label.csv'))
    dataset = CustomDataset(data_dir, labels_file,label_select="Hypertension")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def train_val_dataloader(split,batch_size):
    config = load_config()
    data_dir = config["data_dir"]
    data_dir = os.path.abspath(data_dir)
    data_dir = os.path.join(data_dir,split)
    labels_file = Path(os.path.join(data_dir,'Label.csv'))
    dataset = CustomDataset(data_dir, labels_file,label_select="BP")
    split_dataset = random_split(dataset,[0.8,0.2],generator=torch.Generator().manual_seed(42))
    val_dataset = split_dataset[1]
    train_dataset = split_dataset[0]

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader,val_dataloader

def train_val_dataloader_kfold(split, batch_size, k=5):
    config = load_config()
    data_dir = config["data_dir"]
    data_dir = os.path.abspath(data_dir)
    data_dir = os.path.join(data_dir,split)
    labels_file = Path(os.path.join(data_dir,'Label.csv'))
    dataset = CustomDataset(data_dir, labels_file,label_select="Hypertension")
    kf = KFold(n_splits=k, shuffle=True)
    fold_indices = list(kf.split(dataset)) # type: ignore

    # Create k pairs of dataloaders
    dataloaders = []
    for i in range(k):
        train_indices, val_indices = fold_indices[i]
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        dataloaders.append((train_dataloader, val_dataloader))

    return dataloaders