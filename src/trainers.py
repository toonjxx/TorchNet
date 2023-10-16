import os
from torcheval.metrics.functional import binary_confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np


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
    def __init__(self, model, train_loader , val_loader , num_epochs, learning_rate, log_dir,checkpoint_dir="Checkpoints/temp",early_stop_patience=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.train_loader = train_loader 
        self.val_loader = val_loader 

        self.criterion = nn.BCELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.num_epochs = num_epochs

        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)
        self.checkpoint_dir = checkpoint_dir
        self.early_stop_patience = early_stop_patience

        self.best_val_loss = float('inf')
        self.val_acc = 0
        self.train_acc = 0

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0
            train_acc = 0
            Train_ConfMatrix = torch.zeros(2, 2)
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

                predic = torch.round(output)

                TP = torch.eq(target, predic) & torch.eq(target, torch.ones_like(target))
                TN = torch.eq(target, predic) & torch.eq(target, torch.zeros_like(target))
                FP = torch.ne(target, predic) & torch.eq(target, torch.zeros_like(target))
                FN = torch.ne(target, predic) & torch.eq(target, torch.ones_like(target))
                Train_ConfMatrix[0,0] += TP.sum().item()
                Train_ConfMatrix[1,1] += TN.sum().item()
                Train_ConfMatrix[0,1] += FP.sum().item()
                Train_ConfMatrix[1,0] += FN.sum().item()
                
            train_loss /= len(self.train_loader)
            train_acc,train_f1,train_precision,train_recall,train_sensitivity,train_specificity = self.ConfusionMatrix_Cal(Train_ConfMatrix)
            print(f"Epoch [{epoch + 1}/{self.num_epochs}], Training: Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, Sensitivity: {train_sensitivity:.4f}, Specificity: {train_specificity:.4f}")
            self.writer.add_scalar('train_loss', train_loss, epoch)
            self.writer.add_scalar('train_acc', train_acc , epoch)
            self.train_acc = train_acc

            val_loss,Val_ConfMatrix = self.evaluate()
            val_acc,val_f1,val_precision,val_recall,val_sensitivity,val_specificity = self.ConfusionMatrix_Cal(Val_ConfMatrix)
            print(f"Epoch [{epoch + 1}/{self.num_epochs}], Validation: Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, Sensitivity: {val_sensitivity:.4f}, Specificity: {val_specificity:.4f}")
            self.writer.add_scalar('val_loss', val_loss, epoch)
            self.writer.add_scalar('val_acc', val_acc, epoch)
            self.val_acc = val_acc

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_since_improvement = 0
                checkpoint_path = os.path.join(self.checkpoint_dir, f'model_checkpoint_epoch_{epoch}.pt')
                torch.save(self.model.state_dict(), checkpoint_path)
            else:
                self.epochs_since_improvement += 1        

           # Check if we should stop early
            if self.epochs_since_improvement >= self.early_stop_patience:
                print(f"Validation loss has not improved for {self.early_stop_patience} epochs. Stopping early.")
                break

        return self.train_acc, self.val_acc
    
    def evaluate(self):
        self.model.eval()
        val_loss = 0
        Val_ConfMatrix = torch.zeros(2, 2)
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                val_loss += loss.item()
                predic = torch.round(output)

                TP = torch.eq(target, predic) & torch.eq(target, torch.ones_like(target))
                TN = torch.eq(target, predic) & torch.eq(target, torch.zeros_like(target))
                FP = torch.ne(target, predic) & torch.eq(target, torch.zeros_like(target))
                FN = torch.ne(target, predic) & torch.eq(target, torch.ones_like(target))
                Val_ConfMatrix[0,0] += TP.sum().item()
                Val_ConfMatrix[1,1] += TN.sum().item()
                Val_ConfMatrix[0,1] += FP.sum().item()
                Val_ConfMatrix[1,0] += FN.sum().item()

        val_loss /= len(self.val_loader)

        return val_loss,Val_ConfMatrix
    def ConfusionMatrix_Cal(self,ConfusionMatrix):
        TP = ConfusionMatrix[0,0]
        TN = ConfusionMatrix[1,1]
        FP = ConfusionMatrix[0,1]
        FN = ConfusionMatrix[1,0]
        acc = (TP+TN)/(TP+TN+FP+FN)
        f1 = 2*TP/(2*TP+FP+FN)
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        sensitivity = TP/(TP+FN)
        specificity = TN/(TN+FP)
        return acc,f1,precision,recall,sensitivity,specificity