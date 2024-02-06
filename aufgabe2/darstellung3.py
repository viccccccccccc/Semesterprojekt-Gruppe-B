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
import random
import h5py
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, hdf5_path):
        self.x_len = 7
        self.y_len = 256 * 256
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
            #y = self.scaler_y.transform(y.reshape(-1,1)).flatten()

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


#data = CustomDataset("../../../../../../../../../vol/tmp/feuforsp/rzp-1_sphere1mm_train_2million_bin32.h5")
data = CustomDataset("data2m.h5")
data.normalize()
dl = DataLoader(data, batch_size=1,  num_workers=6, shuffle=True)
model = torch.load("03.02.24, 20:15:01_pca_256x256/model_best.tar", map_location=torch.device('cpu'))
model = model.cpu()
#pca = jl.load("../../../../../../../../../vol/tmp/gruppe_b/pca256.pkl")
pca = jl.load("models/pca256.pkl")
scaler_y = jl.load("scaler_yPCA.joblib")
filter = np.load("nullBild.npz")['name1']

sum=0
while True:
    inputs, d1 = next(iter(dl))
    #dnp = d1.numpy()
    #if not np.any(dnp != 0):
        

    with torch.no_grad():
        inputs = inputs.float()
        inputs = inputs.cpu()
        outputs = model(inputs)
        outputs = outputs.numpy()
        #print(outputs.shape)
        outputs = scaler_y.inverse_transform(outputs.reshape(-1, 1)).flatten()
        #print(outputs.shape)
        d2=pca.inverse_transform(outputs)
    
    d2[d2<0]=0

    #np.savez('nullBild.npz',name1=d1)
    d1 = d1.reshape(256,256)
    d2 = d2.reshape(256,256)

    #test = np.sum(abs(d2 - filter))
    test= np.max(d2)-np.min(d2)
    
    #if(test<200):
    print(test)

    sum+=1

    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(d1)
    axarr[1].imshow(d2)
    plt.show()



