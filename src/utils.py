import yaml
import pandas as pd
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
