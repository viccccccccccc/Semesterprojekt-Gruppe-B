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




def plotte_krasse_sachen(lossestest,lossestrain):
    print(lossestest)
    fig, ax = plt.subplots()
    #ax.set_ylim([max(max(losses['name1']),min(losses['name2'])), max(max(losses['name1']),max(losses['name2']))])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f'Loss Graph')
    ax.plot(lossestrain, '-b',label="Train Losses")
    ax.plot(lossestest, '-g',label="Test Losses")
    ax.legend()

    plt.savefig(f'plots/losses.png')

    plt.figure(1)
    plt.figure(figsize=(10, 10))
    


#files = ["18.01.24, 13:16:29_pca","22.01.24, 20:56:30_pca","24.01.24, 10:54:14_pca","30.01.24, 11:07:54_pca_no_sced","01.02.24, 11:16:06_pca_good_ext"]   #pca

#files = ["23.01.24, 14:29:20_deconv"]  #deconv
#files = ["03.02.24, 20:15:01_pca_256x256"]

files = ["07.02.24, 23:31:48_pca_64x64_clean"]
arraytest = []
arraytrain = []
for fpath in files:
    arraytest = np.concatenate([arraytest,np.load(f'{fpath}/losses.npz')['name2']])
    arraytrain = np.concatenate([arraytrain,np.load(f'{fpath}/losses.npz')['name1']])


plotte_krasse_sachen(arraytest,arraytrain)
