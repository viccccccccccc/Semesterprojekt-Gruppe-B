import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import random


### Hyperparamaters ###
batch_size = 64
anteil_test = 0.2
output_size = 256*256
num_epochs = 10

#######################


class MyDataset(Dataset):
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
        x = torch.tensor(datapoint["X"][:7], dtype=torch.float32)
        y = torch.tensor(datapoint["Y"][:][:], dtype=torch.float32)
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
    return MyDataset(0, split_index, reader), MyDataset(split_index, len(reader), reader)


# Liest die Daten in den DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #init gpu training
print(device)
train_dataset, test_dataset = train_test_split(1./3, "data.h5")
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
model.to(device)                #move model to gpu
criterion = nn.MSELoss()
criterion.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("anfang")

for epoch in range(num_epochs):
    for inputs, labels in train_dataloader:
        labels = labels.view(labels.size(0), -1)    #labels von (256, 256) auf 1D flatten
        inputs,labels =inputs.to(device), labels.to(device)#move data to gpu
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
