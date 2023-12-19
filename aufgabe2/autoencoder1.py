import random
import h5py
import numpy as np
import torch.optim
from torch import nn
from torch.utils.data import Dataset, DataLoader

epochs = 10
lr = 0.001


class CustomDataset(Dataset):
    def __init__(self, min_idx, excluded_max, reader):
        self.reader = reader
        self.min = min_idx
        self.max = excluded_max

    def __len__(self):
        return self.max - self.min

    def __getitem__(self, idx):
        if self.min + idx >= self.max:
            raise Exception("Out of Bounds")
        datapoint = self.reader[self.min + idx]
        x = datapoint["X"][:7]
        y = datapoint["Y"][:][:]
        return x, y


class H5Reader:
    def __init__(self, hdf5_path):
        self.file = h5py.File(hdf5_path, "r")
        self.key_list = list(self.file.keys())
        random.shuffle(self.key_list)

    def __len__(self):
        return len(self.key_list)

    def __getitem__(self, idx):
        if idx >= len(self.key_list):
            raise Exception("Out of Bounds")
        return self.file[self.key_list[idx]]


def train_test_split(anteil_test, hdf5_path):
    reader = H5Reader(hdf5_path)
    split_index = int(len(reader) * (1 - anteil_test))
    return CustomDataset(0, split_index, reader), CustomDataset(split_index, len(reader), reader)


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(int((256 * 256) / 4 ** 0), int((256 * 256) / 4 ** 1)),
            nn.LeakyReLU(),
            nn.Linear(int((256 * 256) / 4 ** 1), int((256 * 256) / 4 ** 2)),
            nn.LeakyReLU(),
            nn.Linear(int((256 * 256) / 4 ** 2), int((256 * 256) / 4 ** 3)),
            nn.LeakyReLU(),
            nn.Linear(int((256 * 256) / 4 ** 3), int((256 * 256) / 4 ** 4)),
        )

        self.decoder = nn.Sequential(
            nn.Linear(int((256 * 256) / 4 ** 4), int((256 * 256) / 4 ** 3)),
            nn.LeakyReLU(),
            nn.Linear(int((256 * 256) / 4 ** 3), int((256 * 256) / 4 ** 2)),
            nn.LeakyReLU(),
            nn.Linear(int((256 * 256) / 4 ** 2), int((256 * 256) / 4 ** 1)),
            nn.LeakyReLU(),
            nn.Linear(int((256 * 256) / 4 ** 1), int((256 * 256) / 4 ** 0)),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train(train_dl):
    model = AutoEncoder()
    loss = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for epoche in range(epochs):
        loss_sum = 0
        counter = 0
        for _, label in train_dl:
            label = label.view(label.size(0), -1).float()
            counter += 1
            reconstructed = model(label)
            loss = loss(reconstructed, label)
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_sum += loss.item()
        print("Epoche: " + str(epoche) + " Loss: " + str(loss_sum / counter))


train_dataset, test_dataset = train_test_split(1. / 3, "data.h5")
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

train(train_dataloader)
