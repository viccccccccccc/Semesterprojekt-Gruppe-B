import os
import random
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, my_reader, key_list):
        self.reader = my_reader
        self.key_list = key_list

    def __len__(self):
        return len(self.key_list)

    def __getitem__(self, idx):
        return self.reader[self.key_list[idx]]


class H5Reader:
    def __init__(self, hdf5_path):
        self.file = h5py.File(hdf5_path, "r")
        self.key_list = list(self.file.keys())
        self.active_normalize = False
        self.x_max = []
        self.y_max = []

    def __len__(self):
        return len(self.key_list)

    def __getitem__(self, idx):
        if isinstance(idx, str):
            datapoint = self.file[idx]
        else:
            datapoint = self.file[self.key_list[idx]]

        x = datapoint["X"][:7]
        y = (datapoint["Y"][:][:]).flatten()
        x_div = x
        y_div = y
        if self.active_normalize:
            x_div = np.divide(x, self.x_max)
            y_div = np.divide(y, self.y_max)
        return x_div, y_div

    def normalize(self):
        x_max, y_max = self.find_max()
        self.active_normalize = True
        for i in range(len(x_max)):
            if x_max[i] == 0:
                x_max[i] = 1
        for j in range(len(y_max)):
            if y_max[j] == 0:
                y_max[j] = 1
        self.x_max = x_max
        self.y_max = y_max


def find_max(h5_reader):
    x_max, y_max = h5_reader[0]
    for i in range(len(h5_reader)):
        x, y = h5_reader[i]
        x_max = [max(ai, bi) for ai, bi in zip(x_max, x)]
        y_max = [max(ai, bi) for ai, bi in zip(y_max, y)]
        print(i)
    return x_max, y_max


def train_test_split(anteil_test, hdf5_path):
    reader = H5Reader(hdf5_path)
    if os.path.exists('max_valuesLoc/x.npz') and os.path.exists('max_valuesLoc/y.npz'):
        x_max = np.load(f'max_valuesLoc/x.npz')
        y_max = np.load(f'max_valuesLoc/y.npz')
        x_max = x_max['name1']
        y_max = y_max['name1']
    else:
        x_max, y_max = find_max(reader)
        np.savez(f'max_valuesLoc/x.npz', name1=x_max)
        np.savez(f'max_valuesLoc/y.npz', name1=y_max)
    print(x_max)
    print(y_max)
    reader.normalize(x_max, y_max)
    split_index = int(len(reader) * (1 - anteil_test))
    return CustomDataset(0, split_index, reader), CustomDataset(split_index, len(reader), reader)


#test_reader = H5Reader("data.h5")
#x_maximum, y_maximum = find_max(test_reader)

#print(len(x_maximum))
#print(x_maximum)
#print(len(y_maximum))
#print(y_maximum)

train_dataset, test_dataset = train_test_split(1. / 3, "data2m.h5")
#train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

