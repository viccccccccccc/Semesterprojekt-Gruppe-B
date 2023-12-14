import datetime
import numpy as np
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
num_epochs = 5
output_every_k = 1
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





class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, 16384),
            nn.ReLU(),
            nn.Linear(16384, output_size)
        )

    def forward(self, x):
        return self.layers(x)






# Training des Modells
def train(train_loader, test_loader):
    train_losses = []
    test_losses = []
    best_model_loss = 1e10
    model = MLP()
    model.to(device)                #move model to gpu
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    last_time = datetime.datetime.now()
    for epoch in range(num_epochs):
        loss_sum = 0
        for inputs, labels in train_loader:
            labels = labels.view(labels.size(0), -1)    #labels von (256, 256) auf 1D flatten
            inputs,labels =inputs.to(device), labels.to(device)#move data to gpu
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()

        # Ausgabe des Losses alle 10 Epochen
        if (epoch + 1) % output_every_k == 0:
            test_loss = 0.0
            model.eval()
            with torch.no_grad():
                for inputs, labels in test_loader:
                    labels = labels.view(labels.size(0), -1)    #labels von (256, 256) auf 1D flatten
                    inputs,labels =inputs.to(device), labels.to(device)#move data to gpu
                    outputs = model(inputs)
                    loss_for_print = criterion(outputs, labels)
                    test_loss += loss_for_print.item()
            avg_train_loss = loss_sum / len(train_loader)
            avg_test_loss = test_loss / len(test_loader)
            train_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)
            now = last_time
            last_time = datetime.datetime.now()
            timediff = last_time - now
            seconds = timediff.total_seconds()
            remainig_seconds = seconds * (num_epochs - epoch) / output_every_k

            if avg_test_loss < best_model_loss:
                best_model_loss = avg_test_loss
                torch.save(model, f'models/model_save_{num_epochs}.tar')

            print(
                f'Epoch: {epoch + 1}/{num_epochs}, Train Loss: {format(avg_train_loss,".8f")},'
                f' Test Loss: {format(avg_test_loss,".8f")}, Learning Rate: {format(optimizer.param_groups[0]["lr"],".8f")},'
                f' Approx. time left: {int(remainig_seconds / 60)} min')
            model.train()
            np.savez(f'losses/losses_{num_epochs}.npz',name1=train_losses,name2=test_losses)
    return train_losses, test_losses



device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu') #init gpu training
print(device)
# Liest die Daten in den DataLoader
train_dataset, test_dataset = train_test_split(1./3, "data.h5")
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print("datensatz geladen und gesplittet!")
print("anfang")
train_losses, test_losses = train(train_dataloader, test_dataloader)
print("Fertig")
