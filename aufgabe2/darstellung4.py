import os
import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib as jl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import h5py
import matplotlib
import optuna

class CustomDataset(Dataset):
    def __init__(self, hdf5_path):
        self.x_len = 7
        self.y_len = 64 * 64
        self.file = h5py.File(hdf5_path, "r")
        self.key_list = list(self.file.keys())
        self.scaler_x = [MinMaxScaler() for _ in range(self.x_len)]
        self.scaler_y = MinMaxScaler()
        self.normalized = False

    def normalize(self):
        if os.path.exists("scaler_xDECON2.joblib") and os.path.exists("scaler_yDECON2.joblib"):
            self.scaler_x = jl.load("scaler_xDECON2.joblib")
            self.scaler_y = jl.load("scaler_yDECON2.joblib")
        else:
            for idx in range(len(self.key_list)):
                datapoint = self.file[self.key_list[idx]]
                x = datapoint["X"][:self.x_len]
                y = (datapoint["Y"][:][:]).flatten()

                # Fit all scales with x and y
                for i in range(self.x_len):
                    self.scaler_x[i].partial_fit(x[i].reshape(-1, 1))
                self.scaler_y.partial_fit(y.reshape(-1, 1))

            jl.dump(self.scaler_x, "scaler_xDECON2.joblib")
            jl.dump(self.scaler_y, "scaler_yDECON2.joblib")

        self.normalized = True

    def denormalize_y(self, y):
        if self.normalized:
            for i in range(self.x_len):
                y[i] = self.scaler_x[i].inverse_transform(y[i].reshape(-1, 1)).flatten()
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
            #y = self.scaler_y.transform(y.reshape(-1,1)).flatten()

        return x, y
    
    def getX(self,idx):
        if isinstance(idx, str):
            datapoint = self.file[idx]
        else:
            datapoint = self.file[self.key_list[idx]]

        x = datapoint["X"][:self.x_len]
        return x

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

model = torch.load("../../../../../../../../../../vol/tmp/gruppe_b/64x64_neu/model_best.tar", map_location=torch.device('cpu'))
model = model.cpu()
pca = jl.load("../../../../../../../../../../vol/tmp/gruppe_b/64x64_neu/pca256_64x64_w0.pkl")
scaler_y = jl.load("../../../../../../../../../../vol/tmp/gruppe_b/64x64_neu/scaler_yPCAneu.joblib")
scaler_x = jl.load("../../../../../../../../../../vol/tmp/gruppe_b/64x64_neu/scaler_xPCAneu.joblib")

data = CustomDataset("../../../../../../../../../vol/tmp/feuforsp/rzp-1_sphere1mm_train_2million_bin32.h5")
dl = DataLoader(data, batch_size=1,  num_workers=6, shuffle=True)




def generate(input):#input: 1 dimensionales np array der laenge 7
    for i in range(len(input)):
        input[i] = input[i].reshape(-1, 1)
        input[i] = scaler_x[i].transform(input[i])
        input[i].flatten()

    torch_data= torch.from_numpy(input)
    with torch.no_grad():
        torch_data = torch_data.float()
        torch_data = torch_data.cpu()
        outputs = model(torch_data)
        outputs = outputs.numpy()

        outputs = scaler_y.inverse_transform(outputs.reshape(-1, 1)).flatten()

        result = pca.inverse_transform(outputs)
        result = result-1
        result[result<0.2]=0
        result = result.reshape(64,64)
        test= np.max(result)-np.min(result)
        if(test<200):
            result = np.zeros((64,64))
        return result
    

for i in range(len(dl)):
    inputs, d1 = next(iter(dl))
        
    d2 = generate(inputs)
    
    d1 = d1.reshape(64,64)

    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(d1)
    axarr[1].imshow(d2)
    plt.show()
