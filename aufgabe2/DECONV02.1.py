import datetime
import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib as jl
import torch
import h5py
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader



#x and y normed 0-1
batch_size = 32 * 8
init_lr = 0.0001
lr_factor = 0.1
lr_patience = 5
kernel_size = 6
kernel_size_2 = 5
stride = 4
stride_2 = 3
padding = 1
num_epochs = 500
save_every_k = 10
test_train_split = 1./5


class CustomDataset(Dataset):
    def __init__(self, hdf5_path):
        self.x_len = 7
        self.y_len = 256 * 256
        self.file = h5py.File(hdf5_path, "r")
        self.key_list = list(self.file.keys())
        self.scalers_x = [MinMaxScaler() for _ in range(self.x_len)]
        self.scaler_y = [MinMaxScaler() for _ in range(self.y_len)]
        self.normalized = False
        

    def normalize(self):
        bs=1000
        if os.path.exists("scaler_xDECON3.joblib") and os.path.exists("scaler_yDECON3.joblib"):
            self.scalers_x = jl.load("scaler_xDECON3.joblib")
            self.scaler_y = jl.load("scaler_yDECON3.joblib")
        else:
            num_batches = int(len(self.key_list) //bs)
            for b in range(num_batches):
                start = b * bs
                end = start + bs
                batch_keys = self.key_list[start:end]
                
                x_batch = []
                y_batch = []
                for key in batch_keys:
                    datapoint = self.file[key]
                    x = datapoint["X"][:self.x_len]
                    y = (datapoint["Y"][:][:]).flatten()
                    x_batch.append(x)
                    y_batch.append(y)

                x_batch = np.array(x_batch)
                y_batch = np.array(y_batch)

                # Fit all scales with x and y
                for i in range(self.x_len):
                    self.scalers_x[i].partial_fit(x_batch[:, i].reshape(-1, 1))
                for i in range(self.y_len):
                    self.scaler_y[i].partial_fit(y_batch[:, i].reshape(-1, 1))

            # If there are any remaining data points that didn't fit into a full batch
            if len(self.key_list) % bs != 0:
                start = num_batches * bs
                batch_keys = self.key_list[start:]
                
                x_batch = []
                y_batch = []
                for key in batch_keys:
                    datapoint = self.file[key]
                    x = datapoint["X"][:self.x_len]
                    y = (datapoint["Y"][:][:]).flatten()
                    x_batch.append(x)
                    y_batch.append(y)

                x_batch = np.array(x_batch)
                y_batch = np.array(y_batch)

                # Fit all scales with x and y
                for i in range(self.x_len):
                    self.scalers_x[i].partial_fit(x_batch[:, i].reshape(-1, 1))
                for i in range(self.y_len):
                    self.scaler_y[i].partial_fit(y_batch[:, i].reshape(-1, 1))

            jl.dump(self.scalers_x, "scaler_xDECON3.joblib")
            jl.dump(self.scaler_y, "scaler_yDECON3.joblib")

        self.normalized = True

    def denormalize_y(self, y):
        if self.normalized:
            for i in range(self.y_len):
                y[i] = self.scaler_y[i].inverse_transform(y[i].reshape(-1, 1)).flatten()
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
                x[i] = self.scalers_x[i].transform(x[i].reshape(-1, 1)).flatten()
            for i in range(self.y_len):
                y[i] = self.scaler_y[i].transform(y[i].reshape(-1, 1)).flatten()

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

class ParameterToImage(nn.Module):
    def __init__(self):
        super(ParameterToImage, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(7, 256),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=kernel_size, stride=stride, padding=padding),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 256, 1, 1)
        x = self.decoder(x)
        x=x.view(x.size(0),-1)
        return x
    
def writeParamFile(file):
    file.write("Params:\n")
    file.write("batch_size = " + str(batch_size) + "\n")
    file.write("init_lr = " + str(init_lr) + "\n")
    file.write("lr_factor = " + str(lr_factor) + "\n")
    file.write("lr_patience = " + str(lr_patience) + "\n")
    file.write("kernel_size = " + str(kernel_size) + "\n")
    file.write("kernel_size_2 = " + str(kernel_size_2) + "\n")
    file.write("stride = " + str(stride) + "\n")
    file.write("stride_2 = " + str(stride_2) + "\n")
    file.write("padding = " + str(padding) + "\n")
    file.write("num_epochs = " + str(num_epochs) + "\n")
    file.write("save_every_k = " + str(save_every_k) + "\n")
    file.write("test_train_split = " + str(test_train_split) + "\n")

def train(model):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=init_lr, weight_decay=0.01)
    schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor, patience=lr_patience)
    last_time = datetime.datetime.now()
    run_directory = last_time.strftime("%d.%m.%y, %H:%M:%S")
    os.mkdir(run_directory)
    file = open(f'{run_directory}/params.txt','w')
    writeParamFile(file)
    file.close()
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

    


device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print(device)
data = CustomDataset("../../../../../../../../../glusterfs/dfs-gfs-dist/feuforsp/rzp-1_sphere1mm_train_2million.h5")
data.normalize()
train_data, test_data = data.split(test_train_split)
train_dataloader = DataLoader(train_data, batch_size=batch_size,  num_workers=72, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size,num_workers=72, shuffle=False)
print("datensatz geladen und gesplittet!")
#model = ParameterToImage()
model = torch.load("17.01.24, 14:11:39/model_best.tar")
train(model)

