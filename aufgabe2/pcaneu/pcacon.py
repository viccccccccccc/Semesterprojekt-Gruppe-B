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
    
data = CustomDataset("../../../../../../../../../../vol/tmp/feuforsp/rzp-1_sphere1mm_train_2million_bin32.h5")



n_components = 256
pca = IncrementalPCA(n_components=n_components,batch_size=256) #
print("init ist fertig")
#for i in range(len(data)//256):
    #subdata=[]
    #if (i*256+256<len(data)):
        #for j in range(256):
            #subdata.append(data[i*256+j]+1)
        #pca.partial_fit(subdata)
#joblib.dump(pca, "pca"+str(n_components)+".pkl")
pca = joblib.load('pca256_64x64_w0.pkl')
print("fit ist fertig")
with h5py.File("pca"+str(n_components)+"_64x64_w0.h5","w") as f:
    for i in range(len(data)):
        grp = f.create_group(data.get_key(i))
        y_pca = pca.transform([data[i]])
        grp.create_dataset("X", data = data.get_x(data.get_key(i)))
        grp.create_dataset("Y", data = y_pca)
        print(i)

print("alles fertig")
