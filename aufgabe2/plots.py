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


# Daten laden und löschen konstanter Spalten




def plotte_krasse_sachen(losses):
    print(losses['name2'])
    fig, ax = plt.subplots()
    #ax.set_ylim([max(max(losses['name1']),min(losses['name2'])), max(max(losses['name1']),max(losses['name2']))])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f'Loss Graph')
    ax.plot(losses['name1'], '-b',label="Train Losses")
    ax.plot(losses['name2'], '-g',label="Test Losses")
    ax.legend()

    plt.savefig(f'plots/losses.png')

    plt.figure(1)
    plt.figure(figsize=(10, 10))
    





plotte_krasse_sachen(np.load(f'10.01.24, 12:36:05/losses.npz'))
