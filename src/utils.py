import yaml
import pandas as pd
import argparse

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
    
