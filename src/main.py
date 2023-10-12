from utils import load_config
import torch
from data_loader import create_data_loader,train_val_dataloader,train_val_dataloader_kfold
from models import ConvNeXt
from trainers import Trainer
from torchsummary import summary

# Load settings from the config file
config = load_config()

# Extract settings from the config
data_dir = config['data_dir']
num_channels = config["num_channels"]
batch_size = config["batch_size"]
k = config["k"]

# Create data loaders
#train_dataloader,val_dataloader = train_val_dataloader("Train", batch_size)
dataloaders = train_val_dataloader_kfold("Train", batch_size, k=5)
test_dataloader = create_data_loader("Test", batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ConvNeXt(depths=[3, 3, 9, 3], dims=[256, 192, 384, 768],in_chans=1,num_classes=1).to(device)
# Create k models and trainers
models = []
trainers = []
for i in range(k):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[256, 192, 384, 768],in_chans=1,num_classes=1).to(device)
    models.append(model)
    trainer = Trainer(model, dataloaders[i], test_dataloader, num_epochs=10, learning_rate=0.001,log_dir=f"logs/fold_{k}")
    trainers.append(trainer)


# Example usage:
if __name__ == "__main__":
    models = []
    trainers = []
    for fold_idx, (train_dataloader, val_dataloader) in enumerate(dataloaders):
        model = ConvNeXt(depths=[3, 3, 9, 3], dims=[256, 192, 384, 768],in_chans=1,num_classes=1).to(device)
        models.append(model)
        trainer = Trainer(model, train_dataloader, val_dataloader, num_epochs=10, learning_rate=0.001,log_dir=f"logs/test")
        trainers.append(trainer)
    for i in range(k):
        print(f"Training model {i+1}/{k}")
        trainers[i].train()

    # Evaluate the models on the test set
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            data, target = data.to(device), target.to(device)
            output = torch.zeros_like(target)
            for i in range(k):
                output += models[i](data).squeeze()
            output /= k
            loss = torch.nn.MSELoss(output, target)
            test_loss += loss.item()
            test_acc += torch.mean(torch.abs(output - target)) # calculate accuracy as MAE
    test_loss /= len(test_dataloader.dataset)
    test_acc /= len(test_dataloader.dataset)*170
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")