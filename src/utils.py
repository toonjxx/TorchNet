import yaml
import pandas as pd
import argparse
import os
import torch
import math


# Load settings from the config file
def load_config(config_file_path="config.yaml"):
    with open(config_file_path, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    return config

def load_sample(file_path):
    try:
        sample = pd.read_csv(file_path,header=None)
        return sample
    
    except Exception as e:
        print(f"Error loading CSV file {file_path}: {str(e)}")
        return None    


def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

# Save model checkpoint
def save_checkpoint(model, optimizer, checkpoint_dir):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    torch.save(state, checkpoint_dir + "/checkpoints.pth.tar")
    print(f"Saving model checkpoint to {checkpoint_dir}")


# Load model checkpoint
def load_checkpoint(checkpoint_path, model, optimizer):
    print("=> Loading checkpoint from '{checkpoint_path}'}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])


# early stopping helper function
class EarlyStopping:
    def __init__(self,patient,mode,metric="acc"):
        self.patient = patient
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.metric = metric
        self.refscore = None

    def __call__(self,score,refscore,model,optimizer,checkpoint_dir):
        if self.mode == "min":
            score = -score

        if self.best_score is None:
            self.best_score = score

        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patient:
                print(f"EarlyStopping! Model does not improve for {self.patient} epochs")
                print(f"Best {self.metric}: {self.best_score:.4f}")
                return True
        else:
            self.best_score = score
            self.refscore = refscore
            save_checkpoint(model, optimizer, checkpoint_dir)
            self.counter = 0
            return False

class MetricMornitor():
    def __init__(self,modeltype = "classification",istrain = True,MinMaxScaler=1):
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
        self.MinMaxScaler = MinMaxScaler
        self.istrain = istrain

    def Batch_Update(self,predic,target):

        if self.modeltype == "classification":
            predic = torch.sigmoid(predic)
            predic = torch.round(predic)

            self.TP += (torch.eq(target, predic) & torch.eq(target, torch.ones_like(target))).sum().item()
            self.TN += (torch.eq(target, predic) & torch.eq(target, torch.zeros_like(target))).sum().item()
            self.FP += (torch.ne(target, predic) & torch.eq(target, torch.zeros_like(target))).sum().item()
            self.FN += (torch.ne(target, predic) & torch.eq(target, torch.ones_like(target))).sum().item()
        elif self.modeltype == "regression":
            self.Accum_AE = self.Accum_AE + torch.sum(torch.abs(predic - target)).item()
            self.num_pred += len(predic)

    def Epoch_Summary(self):
        if self.istrain:
            print("Training:")
        else:
            print("Validation:")
        if self.modeltype == 'classification':
            self.ConfusionMatrix_Cal()
            print(f"Accuracy: {self.Accuracy:.4f}, F1: {self.F1:.4f}, Precision: {self.Precision:.4f}, Recall: {self.Recall:.4f}, Sensitivity: {self.Sensitivity:.4f}, Specificity: {self.Specificity:.4f}")
        elif self.modeltype == 'regression':
            self.Accuracy = self.Accum_AE/self.num_pred    
            print(f"MAE: {self.Accuracy*self.MinMaxScaler:.4f}")
        self.Reset()
    
    def Reset(self):
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.Accum_AE = 0
        self.num_pred = 0
        

    def ConfusionMatrix_Cal(self):
        self.Accuracy = (self.TP+self.TN)/(self.TP+self.TN+self.FP+self.FN)
        self.F1 = 2*self.TP/(2*self.TP+self.FP+self.FN)
        self.Precision = self.TP/(self.TP+self.FP)
        self.Recall = self.TP/(self.TP+self.FN)
        self.Sensitivity = self.TP/(self.TP+self.FN)
        self.Specificity = self.TN/(self.TN+self.FP)


