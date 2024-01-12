import random
from sklearn.preprocessing import MinMaxScaler
import joblib as jl
from torch.utils.data import Dataset
import h5py
import os


class CustomDataset(Dataset):
    def __init__(self, hdf5_path):
        self.x_len = 7
        self.y_len = 256 * 256
        self.file = h5py.File(hdf5_path, "r")
        self.key_list = list(self.file.keys())
        self.scaler_x = [MinMaxScaler() for _ in range(self.x_len)]
        self.scaler_y = [MinMaxScaler() for _ in range(self.y_len)]
        self.normalized = False

    def normalize(self):
        if os.path.exists("scaler_x.joblib") and os.path.exists("scaler_y.joblib"):
            self.scaler_x = jl.load("scaler_x.joblib")
            self.scaler_y = jl.load("scaler_y.joblib")
        else:
            for idx in range(len(self.key_list)):
                datapoint = self.file[self.key_list[idx]]
                x = datapoint["X"][:self.x_len]
                y = (datapoint["Y"][:][:]).flatten()

                # Fit all scales with x and y
                for i in range(self.x_len):
                    self.scaler_x[i].partial_fit(x[i].reshape(-1, 1))
                for i in range(self.y_len):
                    self.scaler_y[i].partial_fit(y[i].reshape(-1, 1))

            jl.dump(self.scaler_x, "scaler_x.joblib")
            jl.dump(self.scaler_y, "scaler_y.joblib")

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
                x[i] = self.scaler_x[i].transform(x[i].reshape(-1, 1)).flatten()
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


# Erstelle ein Custom-Dataset
print("Anfang")
data = CustomDataset("data.h5")
# Normalisiere das Dataset, kann lange dauern, wenn keine scaler vorhanden
data.normalize()
print("Normalisierung fertig")
# Splitte das Dataset mit Testanteil 1/5
train_set, test_set = data.split(1. / 5)
# Denormalizes den Y-Wert von Dataset[0]
data.denormalize_y(data[0][1])
print("fertig")
