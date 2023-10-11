import os
from torch.utils.data import DataLoader, random_split
from custom_dataset import CustomDataset
from utils import load_config
from pathlib import Path
import torch


def create_data_loader(split, batch_size):
    config = load_config()
    data_dir = config["data_dir"]
    data_dir = os.path.abspath(data_dir)
    data_dir = os.path.join(data_dir,split)
    labels_file = Path(os.path.join(data_dir,'Label.csv'))
    dataset = CustomDataset(data_dir, labels_file,label_select="BP")
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
