import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import random
import os

### Hyperparamaters ###
batch_size = 64
anteil_test = 0.2
output_size = 256*256
num_epochs = 50

#######################



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

    def split(self, anteil_test):
        keys = self.key_list.copy()
        random.shuffle(keys)
        split_index = int(len(keys) * anteil_test)
        test = keys[:split_index]
        train = keys[split_index:]

        return CustomDataset(self, train), CustomDataset(self, test)

    def find_max(self):
        if os.path.exists('max_values/x.npz') and os.path.exists('max_values/y.npz'):
            x_max = np.load(f'max_values/x.npz')
            y_max = np.load(f'max_values/y.npz')
            x_max = x_max['name1']
            y_max = y_max['name1']
        else:
            x_max, y_max = self[0]
            for i in range(len(self)):
                x, y = self[i]
                x_max = [max(ai, bi) for ai, bi in zip(x_max, x)]
                y_max = [max(ai, bi) for ai, bi in zip(y_max, y)]
            np.savez(f'max_values/x.npz', name1=x_max)
            np.savez(f'max_values/y.npz', name1=y_max)
        return x_max, y_max




device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)
reader = H5Reader("data.h5")
reader.normalize()
train_dataset, test_dataset = reader.split(1. / 3)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size,  num_workers=72, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size,num_workers=72, shuffle=False)
print("datensatz geladen und gesplittet!")

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(7, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        return self.layers(x)

model = MLP()
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("anfang")

for epoch in range(num_epochs):
    for inputs, labels in train_dataloader:
        inputs = inputs.float()
        labels = labels.float()
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).cpu()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
