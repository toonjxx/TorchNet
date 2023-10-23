from utils import load_config
import torch
from datasets import create_data_loader,train_val_dataloader_kfold
from models import ConvNeXt
from trainers import Trainer_Hyper, Trainer_BP
import torch.optim.lr_scheduler
import time


# Load settings from the config file
config = load_config()

# Extract settings from the config
data_dir = config['data_dir']
num_channels = config["num_channels"]
batch_size = config["batch_size"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def crossvalidation():
    k = config["k"]
    dataloaders = train_val_dataloader_kfold("Train", batch_size, k=5)
    models = []
    trainers = []
    timenow = time.time()
    for fold_idx, (train_dataloader, val_dataloader) in enumerate(dataloaders):
        model = ConvNeXt(depths=[3, 3, 27, 3], dims=[32, 64, 128, 256],in_chans=1,num_classes=num_channels,headtype="sigmoid",drop_path_rate=0.5).to(device)
        models.append(model)
        checkpoint_dir = f"Checkpoints/val_{timenow}/fold{fold_idx}"
        log_dir = f"logs/val_{timenow}/fold{fold_idx}"
        trainer = Trainer_Hyper(model, train_dataloader, val_dataloader, num_epochs=200 ,log_dir=log_dir,checkpoint_dir=checkpoint_dir,early_stop_patience=20)
        trainers.append(trainer)
    val_acc = []
    train_acc = []
    for i in range(k):
        print(f"Training model {i+1}/{k}")
        train,val=trainers[i].train()
        val_acc.append(val)
        train_acc.append(train)
    print(f"Average train accuracy: {sum(train_acc)/k}")
    print(f"Average val accuracy: {sum(val_acc)/k}")

def train():
    timenow = time.time()
    test_dataloader = create_data_loader("Test", batch_size)
    train_dataloader = create_data_loader("Train", batch_size)
    model_name = 'ConvNeXt1d_L_Hypertension'
    checkpoint_dir = f"Checkpoints/train_{model_name}_{timenow}"
    log_dir = f"logs/train_{model_name}_{timenow}"

    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],in_chans=1,num_classes=num_channels,headtype="sigmoid").to(device)
    trainer = Trainer_Hyper(model, train_dataloader, test_dataloader, num_epochs=500,log_dir=log_dir,checkpoint_dir=checkpoint_dir,early_stop_patience=50)
    trainer.train()

if __name__ == "__main__":
    crossvalidation()
    #train()