from llava.train.train import train
import numpy as np
import torch
import os
import random

def set_seed(seed=42):
    """random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)



if __name__ == "__main__":
    set_seed(42)
    train()
