import os
import random
import h5py
import numpy as np

INfilename = "data.h5"
OUTfilename = "dataOUT.h5"

class H5Reader:
    def __init__(self, hdf5_path):
        self.file = h5py.File(hdf5_path, "r")
        self.key_list = list(self.file.keys())
        self.active_normalize = False
        self.x_max = []
        self.y_max = []
        random.shuffle(self.key_list)

    def __len__(self):
        return len(self.key_list)

    def __getitem__(self, idx):
        if idx >= len(self.key_list):
            raise Exception("Out of Bounds")
        datapoint = self.file[self.key_list[idx]]
        x = datapoint["X"][:7]
        y = (datapoint["Y"][:][:]).flatten()
        x_div = x
        y_div = y
        if self.active_normalize:
            print("Let's normalize")
            x_div = np.divide(x, self.x_max)
            y_div = np.divide(y, self.y_max)
        return x_div, y_div

    def normalize(self, x_max, y_max):
        self.active_normalize = True
        for i in range(len(x_max)):
            if x_max[i] == 0:
                x_max[i] = 1
        for j in range(len(y_max)):
            if y_max[j] == 0:
                y_max[j] = 1
        self.x_max = x_max
        self.y_max = y_max

def comprimiern():
    if not(os.path.exists('max_values/x.npz') and os.path.exists('max_values/y.npz')):
        find_max()

    reader = H5Reader(INfilename)
    
    y_max = np.load(f'max_values/y.npz')
    x_max = np.load(f'max_values/x.npz')
    x_max = x_max['name1']
    y_max = y_max['name1']
    y_mask = y_max > 0
    yCOMPMAX = np.compress(y_mask,y_max)

    reader.normalize(x_max,y_max)
    

    fileIN = h5py.File(INfilename)
    keys = list(fileIN.keys())
    keys.sort(key=int)
    keys = keys[:20]
    print(keys)
    with h5py.File(OUTfilename, 'w') as fileOUT:
        for key in keys:
            element = fileIN[key]
            xIN = element["X"]
            yIN = element["Y"]
            yIN = yIN[:][:].flatten()
            xOUT = xIN[:7]
            xOUT = np.divide(xOUT,x_max)

            #xOUT,yOUT = reader[int(key)]

            #yOUT = yIN[:,y_mask]
            yOUT = np.compress(y_mask,yIN)
            yOUT = np.divide(yOUT,yCOMPMAX)

            
            group = fileOUT.create_group(key)
            group.create_dataset("X",data=xOUT)
            group.create_dataset("Y",data=yOUT)

def erweitern(dataIN):
    if os.path.exists('max_values/x.npz') and os.path.exists('max_values/y.npz'):
        y_max = np.load(f'max_values/y.npz')
        x_max = np.load(f'max_values/x.npz')
        x_max = x_max['name1']
        y_max = y_max['name1']
        y_mask = y_max > 0
        yCOMPMAX = np.compress(y_mask,y_max)
        #dataIN = np.multiply(dataIN,yCOMPMAX)

        dataOUT=[]
        iterator = 0
        for i in y_max:
            if i == 0:
                dataOUT.append(0)
            else:
                dataOUT.append(dataIN[iterator])
                iterator += 1

def find_max():
    fileIN = h5py.File(INfilename)
    keys = list(fileIN.keys())
    x_max = fileIN[keys[0]]["X"]
    y_max = fileIN[keys[0]]["Y"]
    




    x_max, y_max = h5_reader[0]
    for i in range(len(h5_reader)):
        x, y = h5_reader[i]
        x_max = [max(ai, bi) for ai, bi in zip(x_max, x)]
        y_max = [max(ai, bi) for ai, bi in zip(y_max, y)]
        print(i)
    return x_max, y_max


comprimiern()