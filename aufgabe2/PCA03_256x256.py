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
num_epochs = 1000
save_every_k = 10
init_lr = 0.000001
test_train_split = 1./5

class CustomDataset(Dataset):
    def __init__(self, hdf5_path):
        self.x_len = 7
        self.y_len = 256
        self.file = h5py.File(hdf5_path, "r")
        self.key_list = list(self.file.keys())
        self.scaler_x = [MinMaxScaler() for _ in range(self.x_len)]
        self.scaler_y = MinMaxScaler()
        self.normalized = False

    def normalize(self):
        if os.path.exists("scaler_xPCA.joblib") and os.path.exists("scaler_yPCA.joblib"):
            self.scaler_x = jl.load("scaler_xPCA.joblib")
            self.scaler_y = jl.load("scaler_yPCA.joblib")
        else:
            for idx in range(len(self.key_list)):
                datapoint = self.file[self.key_list[idx]]
                x = datapoint["X"][:self.x_len]
                y = (datapoint["Y"][:][:]).flatten()

                # Fit all scales with x and y
                for i in range(self.x_len):
                    self.scaler_x[i].partial_fit(x[i].reshape(-1, 1))
                self.scaler_y.partial_fit(y.reshape(-1, 1))

            jl.dump(self.scaler_x, "scaler_xPCA.joblib")
            jl.dump(self.scaler_y, "scaler_yPCA.joblib")

        self.normalized = True

    def denormalize_y(self, y):
        if self.normalized:
            for i in range(self.y_len):
                y[i] = self.scaler_y.inverse_transform(y[i].reshape(-1, 1)).flatten()
        return y

    def __len__(self):
        return len(self.key_list)

    def __getitem__(self, idx):
        if isinstance(idx, str):
            datapoint = self.file[idx]
        else:
            datapoint = self.file[self.key_list[idx]]

        x = datapoint["X"][:self.x_len]
        y = (datapoint["Y"][:][:]).flatten()

        # Normalize each group in x and y
        if self.normalized:
            for i in range(self.x_len):
                x[i] = self.scaler_x[i].transform(x[i].reshape(-1, 1)).flatten()
            y = self.scaler_y.transform(y.reshape(-1,1)).flatten()

        return x, y

    def split(self, anteil_test):
        keys = self.key_list.copy()
        random.shuffle(keys)
        split_index = int(len(keys) * anteil_test)
        test_keys = keys[:split_index]
        train_keys = keys[split_index:]

        return CustomView(self, train_keys), CustomView(self, test_keys)
    
class CustomView(Dataset):
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
            nn.Linear(2048, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 256),
    
        )

    def forward(self, x):
        return self.fc(x)

def writeParamFile(file):
    file.write("Params:\n")
    file.write("batch_size = " + str(batch_size) + "\n")
    file.write("init_lr = " + str(init_lr) + "\n")
    file.write("num_epochs = " + str(num_epochs) + "\n")
    file.write("save_every_k = " + str(save_every_k) + "\n")
    file.write("test_train_split = " + str(test_train_split) + "\n")

def train(model):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = init_lr)
    #schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')
    last_time = datetime.datetime.now()
    run_directory = last_time.strftime("%d.%m.%y, %H:%M:%S")
    run_directory += "_pca_256x256"
    os.mkdir(run_directory)
    file = open(f'{run_directory}/params.txt','w')
    writeParamFile(file)
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
            #schedular.step(loss)
        #loss_sum = loss_sum.cpu()
        
        now = last_time
        last_time = datetime.datetime.now()
        timediff = last_time - now
        minutes = timediff.total_seconds()/60
        remainig_minutes = minutes * (num_epochs - epoch)

        print(f'Epoch {epoch+1} from {num_epochs}, Loss: {loss.item():.8f}, Learning Rate: {optimizer.param_groups[0]["lr"]:.8f}, Aprox. Time left: {remainig_minutes :.1f}min')

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


device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print(device)
data = CustomDataset("pca256_256x256.h5")
data.normalize()
train_data, test_data = data.split(test_train_split)
train_dataloader = DataLoader(train_data, batch_size=batch_size,  num_workers=72, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size,num_workers=72, shuffle=False)
print("datensatz geladen und gesplittet!")
model = MLP()
#model = torch.load("30.01.24, 11:07:54_pca_no_sced/model_best.tar")
train(model)
