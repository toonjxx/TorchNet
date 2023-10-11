import os
from torch.utils.data import DataLoader
from custom_dataset import CustomDataset
from utils import load_config
from pathlib import Path

def create_data_loader(split, batch_size):
    config = load_config()
    data_dir = config["data_dir"]
    data_dir = os.path.abspath(data_dir)
    data_folder = os.path.join(data_dir,split)
    csv_files = [os.path.join(data_folder, file) for file in data_folder if file.endswith(".csv")]

    labels_file = Path(os.path.join(data_folder,'Label.csv'))

    dataset = CustomDataset(csv_files, labels_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader