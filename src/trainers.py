import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
import time
from torch.optim.lr_scheduler import StepLR
from utils import EarlyStopping
    

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
    def __init__(self, model, train_loader , val_loader , num_epochs, lr = 0.001,optimizer = None, log_dir=None,checkpoint_dir=None,early_stop_patience=15,lr_scheduler = None,modeltype = "classification"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.modeltype = modeltype
        self.train_loader = train_loader 
        self.val_loader = val_loader 

        self.criterion = nn.BCEWithLogitsLoss()
        self.lr = lr
        if optimizer is None:
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        if lr_scheduler is None:
            self.lr_scheduler = StepLR(self.optimizer, step_size=10, gamma=0.8)
        self.num_epochs = num_epochs

        self.log_dir = log_dir
        self.train_writer = SummaryWriter(log_dir=f'{log_dir}/train')
        self.val_writer = SummaryWriter(log_dir=f'{log_dir}/val')
        self.checkpoint_dir = checkpoint_dir
        self.early_stop_patience = early_stop_patience

        self.best_val_acc = 0
        self.best_train_acc = 0
        self.val_acc = 0
        self.train_acc = 0
        self.epochs_since_improvement = 0

        self.train_MetricMornitor = MetricMornitor(modeltype=self.modeltype)
        self.val_MetricMornitor = MetricMornitor(modeltype=self.modeltype)


    def train(self):
        earlyStop = EarlyStopping(patient=self.early_stop_patience,mode='max')

        for epoch in range(self.num_epochs):

            start_time = time.time()
            self.model.train()
            train_loss = 0

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)

                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                self.train_MetricMornitor.Batch_Update(output,target)

            self.train_MetricMornitor.Epoch_Summary()


            val_loss = self.evaluate()
            self.val_MetricMornitor.Epoch_Summary()
            stop = time.time()

            print(f"Epoch [{epoch + 1}/{self.num_epochs}] - Learning rate {self.lr_scheduler.get_last_lr()} - Processing times: {stop-start_time:.2f} seconds")


            '''
            print(f"Training: Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, Sensitivity: {train_sensitivity:.4f}, Specificity: {train_specificity:.4f}")
            print(f"Validation: Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, Sensitivity: {val_sensitivity:.4f}, Specificity: {val_specificity:.4f}")
            '''
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            if earlyStop(val_acc):
                


            if earlyStop.early_stop:
                print("Early stopping")
                break

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_train_acc = train_acc
                self.epochs_since_improvement = 0
                if self.checkpoint_dir is not None:
                    checkpoint_path = os.path.join(self.checkpoint_dir, f'Checkpoints.pt')
                    print(f"Saving model checkpoint to {checkpoint_path}")
                    if not os.path.exists(self.checkpoint_dir):
                        os.makedirs(self.checkpoint_dir)
                    torch.save(self.model.state_dict(), checkpoint_path)
            else:
                self.epochs_since_improvement += 1        

           # Check if we should stop early
            if self.epochs_since_improvement >= self.early_stop_patience:
                print(f"Validation loss has not improved for {self.early_stop_patience} epochs. Stopping early.")
                break

        return self.best_train_acc, self.best_val_acc
    
    def evaluate(self):
        self.model.eval()
        val_loss = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                val_loss += loss.item()

                self.val_MetricMornitor.Batch_Update(output,target)

        self.val_MetricMornitor.Epoch_Summary()

        val_loss /= len(self.val_loader)
        return val_loss
    
    
class MetricMornitor():
    def __init__(self,modeltype = "classification"):
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.Accuracy = 0
        self.Precision = 0
        self.Recall = 0
        self.Sensitivity = 0
        self.Specificity = 0
        self.F1 = 0
        self.modeltype = modeltype
        self.Accum_AE = 0
        self.num_pred = 0


    def Batch_Update(self,predic,target):
        if self.modeltype == "classification":
            predic = torch.sigmoid(predic)
            predic = torch.round(predic)
            self.TP += (torch.eq(target, predic) & torch.eq(target, torch.ones_like(target))).sum().item()
            self.TN += (torch.eq(target, predic) & torch.eq(target, torch.zeros_like(target))).sum().item()
            self.FP += (torch.ne(target, predic) & torch.eq(target, torch.zeros_like(target))).sum().item()
            self.FN += (torch.ne(target, predic) & torch.eq(target, torch.ones_like(target))).sum().item()
        elif self.modeltype == "regression":
            self.Accum_AE += torch.abs(predic - target)
            self.num_pred += len(predic)

    def Epoch_Summary(self):
        if self.modeltype == 'classification':
            

    def ConfusionMatrix_Cal(self):
        self.acc = (self.TP+self.TN)/(self.TP+self.TN+self.FP+self.FN)
        self.f1 = 2*self.TP/(2*self.TP+self.FP+self.FN)
        self.precision = self.TP/(self.TP+self.FP)
        self.recall = self.TP/(self.TP+self.FN)
        self.sensitivity = self.TP/(self.TP+self.FN)
        self.specificity = self.TN/(self.TN+self.FP)

    
