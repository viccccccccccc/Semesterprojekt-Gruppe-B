import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
import  matplotlib.pyplot as plt
import matplotlib
import datetime
matplotlib.use("Agg")

import h5py
filename = "data100k.h5"

with h5py.File(filename, "r") as f:
    
    dset = f['1']
    print(dset)
    print(dset.shape)
    print(dset.dtype)
    


f.close()
