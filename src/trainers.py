import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter

class Trainer_BP:
    def __init__(self, model, train_dataset, val_dataset, num_epochs, learning_rate, log_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_dataset
        self.val_loader = val_dataset
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.num_epochs = num_epochs
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0
            train_acc = 0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                train_acc += torch.mean(torch.abs(output - target))

            train_loss /= len(self.train_loader)
            train_acc /= len(self.train_loader)
            print(f"Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc*140:.4f}")

            self.model.eval()
            val_loss = 0
            val_acc = 0
            with torch.no_grad():
                for data, target in self.val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    val_loss += loss.item()
                    val_acc += torch.mean(torch.abs(output - target))
            val_loss /= len(self.val_loader)
            val_acc /= len(self.val_loader)
            print(f"Epoch [{epoch + 1}/{self.num_epochs}], Val Loss: {val_loss:.4f}, Val Acc: {val_acc*140:.4f}")

class Trainer_Hyper:
    def __init__(self, model, train_dataset, val_dataset, num_epochs, learning_rate, log_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_dataset
        self.val_loader = val_dataset
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.num_epochs = num_epochs
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0
            train_acc = 0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                train_acc += (output.argmax(dim=1) == target).float().mean()

            train_loss /= len(self.train_loader)
            train_acc /= len(self.train_loader)
            print(f"Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

            self.model.eval()
            val_loss = 0
            val_acc = 0
            with torch.no_grad():
                for data, target in self.val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    val_loss += loss.item()
                    train_acc += (output.argmax(dim=1) == target).float().mean()
            val_loss /= len(self.val_loader)
            val_acc /= len(self.val_loader)
            print(f"Epoch [{epoch + 1}/{self.num_epochs}], Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")    