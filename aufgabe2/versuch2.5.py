import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import random
import os

### Hyperparamaters ###
batch_size = 2
anteil_test = 0.2
output_size = 256*256
num_epochs = 50
save_every_k = 10
test_train_split = 1./5
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






class DeconvNet(nn.Module):
    def __init__(self):
        super(DeconvNet, self).__init__()
        self.fc = nn.Linear(7, 256)  # Fully connected layer
        self.relu = nn.ReLU()  # Activation function
        self.deconv1 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1) # Deconvolution layer
        self.deconv2 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1)  # Deconvolution layer
        self.deconv3 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1)  # Deconvolution layer
        self.deconv4 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1)  # Deconvolution layer

    def forward(self, x):
        x = self.relu(self.fc(x))
        x = x.view(x.size(0), 1, 16, 16)  # reshape to match the shape needed for the first deconvolution layer
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.relu(self.deconv3(x))
        x = self.deconv4(x)
        #for element in x: element.flatten()
        x=x.view(x.size(0),output_size)
        return x
    

def train():
    model = DeconvNet()
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    last_time = datetime.datetime.now()
    run_directory = last_time.strftime("%d.%m.%y, %H:%M:%S")
    os.mkdir(run_directory)
    train_losses = []
    test_losses = []
    best_model_loss = 1e10

    print("anfang")
    
    for epoch in range(num_epochs):
        loss_sum = 0
        for inputs, labels in train_dataloader:
            inputs = inputs.float()
            labels = labels.float()
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).cpu()
            loss = criterion(outputs, labels)
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()

        
        now = last_time
        last_time = datetime.datetime.now()
        timediff = last_time - now
        minutes = timediff.total_seconds()/60
        remainig_minutes = minutes * (num_epochs - epoch)

        print(f'Epoch {epoch+1} from {num_epochs}, Loss: {loss.item()}, Aprox. Time left: {remainig_minutes}')

        if((epoch+1)%save_every_k==0):
            test_loss = 0.0
            model.eval()
            with torch.no_grad():
                for inputs, labels in test_dataloader:
                    inputs,labels =inputs.to(device), labels.to(device)#move data to gpu
                    outputs = model(inputs)
                    loss_for_print = criterion(outputs, labels)
                    test_loss += loss_for_print.item()
            avg_train_loss = loss_sum / len(train_dataloader)
            avg_test_loss = test_loss / len(test_dataloader)
            train_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)

            if avg_test_loss < best_model_loss:
                best_model_loss = avg_test_loss
                torch.save(model, f'{run_directory}/model_best.tar')
            torch.save(model, f'{run_directory}/model_{epoch+1}.tar')
    np.savez(f'{run_directory}/losses.npz',name1=train_losses,name2=test_losses)


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)
reader = H5Reader("data.h5")
reader.normalize()
train_dataset, test_dataset = reader.split(test_train_split)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size,  num_workers=72, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size,num_workers=72, shuffle=False)
print("datensatz geladen und gesplittet!")

train()
