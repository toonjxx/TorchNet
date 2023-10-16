from utils import load_config
import torch
from data_loader import create_data_loader,train_val_dataloader,train_val_dataloader_kfold
from models import ConvNeXt
from trainers import Trainer_Hyper, Trainer_BP
from torchsummary import summary

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
    for fold_idx, (train_dataloader, val_dataloader) in enumerate(dataloaders):
        model = ConvNeXt(depths=[2, 2, 6, 2], dims=[96, 192, 384, 768],in_chans=1,num_classes=1).to(device)
        models.append(model)
        trainer = Trainer_Hyper(model, train_dataloader, val_dataloader, num_epochs=30, learning_rate=0.001,log_dir=f"logs/test")
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
    test_dataloader = create_data_loader("Test", batch_size)
    train_dataloader = create_data_loader("Train", batch_size)
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],in_chans=1,num_classes=1).to(device)
    trainer = Trainer_Hyper(model, train_dataloader, test_dataloader, num_epochs=30, learning_rate=0.001,log_dir=f"logs/temp",checkpoint_dir="Checkpoints/temp",early_stop_patience=5)
    trainer.train()

if __name__ == "__main__":
    crossvalidation()
    #train()