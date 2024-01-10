import os
from sklearn import datasets
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import joblib as jl
import h5py


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
    
    def get_key(self, idx):
        return self.key_list[idx]


print("Anfang")
data = CustomDataset("../../../../../../../../../glusterfs/dfs-gfs-dist/gruppe_b/pca256.h5")
data.normalize()
print("Normalisierung Fertig")

with h5py.File("pca256_2m_comp_norm.h5","w") as f:
    for i in range(len(data)):
        x,y = data[i]
        grp = f.create_group(data.get_key(i))
        grp.create_dataset("X", data = x)
        grp.create_dataset("Y", data = y)

print("Speichern Fertig")


