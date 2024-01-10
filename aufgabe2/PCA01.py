import random
from sklearn.preprocessing import MinMaxScaler
import h5py
from torch.utils.data import Dataset
import joblib as jl
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import os
from sklearn.decomposition import IncrementalPCA
import joblib
import datetime
import numpy as np

batch_size = 64
num_epochs = 500
save_every_k = 10
init_lr = 0.001
test_train_split = 1./5

class CustomDataset(Dataset):
    def __init__(self, hdf5_path):
        self.file = h5py.File(hdf5_path, "r")
        self.key_list = list(self.file.keys())
        self.scaler_x = [MinMaxScaler() for _ in range(7)]
        self.scaler_y = [MinMaxScaler() for _ in range(256)]
        self.normalized = False

    def normalize(self):
        if  os.path.exists("scaler_x.joblib") and os.path.exists("scaler_y.joblib"):
            self.scaler_x = jl.load("scaler_x.joblib")
            self.scaler_y = jl.load("scaler_y.joblib")
        else:
            for idx in range(len(self.key_list)):
                datapoint = self.file[self.key_list[idx]]
                x = datapoint["X"][:7]
                y = (datapoint["Y"][:][:]).flatten()

                # Fit athe scalers with x and y
                for i in range(7):
                    self.scaler_x[i].partial_fit(x[i].reshape(-1, 1))
                for i in range(256):
                    self.scaler_y[i].partial_fit(y[i].reshape(-1, 1))

            jl.dump(self.scaler_x, "scaler_x.joblib")
            jl.dump(self.scaler_y, "scaler_y.joblib")

        self.normalized = True

    def denormalize(self, y):
        if self.normalized:
            for i in range(256):
                y[i] = self.scaler_y[i].inverse_transform(y[i].reshape(-1, 1)).flatten()
        return y


    def __len__(self):
        return len(self.key_list)

    def __getitem__(self, idx):
        if isinstance(idx, str):
            datapoint = self.file[idx]
        else:
            datapoint = self.file[self.key_list[idx]]

        x = datapoint["X"][:7]
        y = (datapoint["Y"][:][:]).flatten()

        # Normalize each group in x and y
        if  self.normalized:
            for i in range(7):
                x[i] = self.scaler_x[i].transform(x[i].reshape(-1, 1)).flatten()
            for i in range(256):
                y[i] = self.scaler_y[i].transform(y[i].reshape(-1, 1)).flatten()

        return x, y
    
    def split(self, anteil_test):
        keys = self.key_list.copy()
        random.shuffle(keys)
        split_index = int(len(keys) * anteil_test)
        test = keys[:split_index]
        train = keys[split_index:]

        return CustomDataset2(self, train), CustomDataset2(self, test)
    
class CustomDataset2(Dataset):
    def __init__(self, my_reader, key_list):
        self.reader = my_reader
        self.key_list = key_list

    def __len__(self):
        return len(self.key_list)

    def __getitem__(self, idx):
        return self.reader[self.key_list[idx]]
    
    


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(7, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 8192),
            nn.LeakyReLU(),
            nn.Linear(8192, 32768),
            nn.LeakyReLU(),
            nn.Linear(32768, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 256),
    
        )

    def forward(self, x):
        return self.fc(x)

def train():
    model = MLP()
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = init_lr)
    schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')
    last_time = datetime.datetime.now()
    run_directory = last_time.strftime("%d.%m.%y, %H:%M:%S")
    os.mkdir(run_directory)
    train_losses = []
    test_losses = []
    best_model_loss = 1e10

    print("anfang")
    
    for epoch in range(num_epochs):
        loss_sum = 0
        #loss_sum = loss_sum.to(device)
        for inputs, labels in train_dataloader:
            #print(inputs.size())
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.float()
            labels = labels.float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()
            schedular.step(loss)
        #loss_sum = loss_sum.cpu()
        
        now = last_time
        last_time = datetime.datetime.now()
        timediff = last_time - now
        minutes = timediff.total_seconds()/60
        remainig_minutes = minutes * (num_epochs - epoch)

        print(f'Epoch {epoch+1} from {num_epochs}, Loss: {loss.item()}, Aprox. Time left: {remainig_minutes}min')

        if((epoch+1)%save_every_k==0):
            test_loss = 0.0
            model.eval()
            with torch.no_grad():
                for inputs, labels in test_dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)#move data to gpu
                    labels = labels.float()
                    inputs = inputs.float()
                    outputs = model(inputs)
                    loss_for_print = criterion(outputs, labels)
                    loss_for_print = loss_for_print.cpu()
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


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
data = CustomDataset("../../../../../../../../../glusterfs/dfs-gfs-dist/gruppe_b/pca256.h5")
data.normalize()
train_data, test_data = data.split(test_train_split)
train_dataloader = DataLoader(train_data, batch_size=batch_size,  num_workers=72, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size,num_workers=72, shuffle=False)
print("datensatz geladen und gesplittet!")
train()
