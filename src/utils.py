import yaml
import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    


    
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:,  None] * x + self.bias[:, None]
            return x
