import numpy as np
import h5py
from torch.utils.data import Dataset
from sklearn.decomposition import IncrementalPCA
import joblib

class CustomDataset(Dataset):
    def __init__(self, hdf5_path):
        self.file = h5py.File(hdf5_path, "r")
        self.key_list = list(self.file.keys())

    def __len__(self):
        return len(self.key_list)

    def __getitem__(self, idx):
        datapoint = self.file[self.key_list[idx]]
        y = (datapoint["Y"][:][:]).flatten()
        return y
    def get_key(self, idx):
        return self.key_list[idx]
    
    def rem_key(self, key):
        self.key_list.remove(key)
    
    def get_x(self, key):
        datapoint = self.file[key]
        return datapoint["X"][:]
    
data = CustomDataset("data2m.h5")

nullen = []
for i in range(len(data)):
    y = data[i]
    if not np.any(y != 0):
        key = data.get_key(i)
        nullen.append(key)

for key in nullen:
    data.rem_key(key)

n_components = 256
pca = IncrementalPCA(n_components=n_components,batch_size=256) #
print("init ist fertig")
pca.fit(data)
joblib.dump(pca, "pca"+str(n_components)+".pkl")
#ipca = joblib.load('ipca.pkl')
print("fit ist fertig")
with h5py.File("pca"+str(n_components)+".h5","w") as f:
    for i in range(len(data)):
        grp = f.create_group(data.get_key(i))
        y_pca = pca.transform([data[i]])
        grp.create_dataset("X", data = data.get_x(data.get_key(i)))
        grp.create_dataset("Y", data = y_pca)

print("alles fertig")
