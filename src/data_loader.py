import os
from torch.utils.data import DataLoader
from custom_dataset import CustomDataset
from utils import load_config
from pathlib import Path

def create_data_loader(split, batch_size):
    config = load_config()
    data_dir = config["data_dir"]
    data_dir = os.path.abspath(data_dir)
    data_dir = os.path.join(data_dir,split)
    labels_file = Path(os.path.join(data_dir,'Label.csv'))
    dataset = CustomDataset(data_dir, labels_file,label_select="BP")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader