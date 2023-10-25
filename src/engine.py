import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
import time
from torch.optim.lr_scheduler import StepLR
from utils import EarlyStopping, MetricMornitor

class Engine:
    def __init__(self, model, train_loader , val_loader, optimizer = None, log_dir = None, checkpoint_dir=None,lr_scheduler = None,args=None):
        self.train_loader = train_loader 
        self.val_loader = val_loader

        if args is None:
            raise ValueError("args is None")
        
        self.args = args

        if args.modeltype == "classification":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.MSELoss()

        self.device = args.device
        self.model = model.to(self.device)

        if checkpoint_dir is None:
            self.checkpoint_dir = self.args.checkpoint_dir
        self.checkpoint_dir = checkpoint_dir

        if log_dir is None:
            self.log_dir = self.args.log_dir
        self.log_dir = log_dir

        if optimizer is None:
            self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr)
        if lr_scheduler is None:
            self.lr_scheduler = StepLR(self.optimizer, step_size=10, gamma=0.8)

        self.train_writer = SummaryWriter(log_dir=f'{self.log_dir}/train')
        self.val_writer = SummaryWriter(log_dir=f'{self.log_dir}/val')


        self.train_MetricMornitor = MetricMornitor(modeltype=args.modeltype,istrain=True)
        self.val_MetricMornitor = MetricMornitor(modeltype=args.modeltype,istrain=False)


    def train(self):
        if self.args.modeltype == "classification":
            metric = "Accuracy"
        else:
            metric = "MAE"
        earlyStop = EarlyStopping(patient=self.args.early_stop_patience,mode='max',metric=metric)

        for epoch in range(self.args.num_epochs):
            print(f"Epoch [{epoch + 1}/{self.args.num_epochs}] - Learning rate {self.lr_scheduler.get_last_lr()}")
            start_time = time.time()

            self.model.train(True)
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

            self.evaluate()
            stop = time.time()

            print(f"Processing times: {stop-start_time:.2f} seconds")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            if earlyStop(self.val_MetricMornitor.Accuracy,self.train_MetricMornitor.Accuracy,self.model,self.optimizer,self.checkpoint_dir):
                break

        return earlyStop.refscore ,earlyStop.best_score
    
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
    
    

    
