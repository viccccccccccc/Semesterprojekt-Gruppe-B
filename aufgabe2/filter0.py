import os
import h5py
import numpy as np


if os.path.exists('max_values/x.npz') and os.path.exists('max_values/y.npz'):
    y_max = np.load(f'max_values/y.npz')
    x_max = np.load(f'max_values/x.npz')
    y_mask = y_max > 0
    yCOMPMAX = np.compress(y_mask,y_max)
    fileIN = h5py.File("data.h5")
    keys = list(fileIN.keys())
    keys = keys[:20]
    with h5py.File('dataOUT.h5', 'w') as fileOUT:
        for key in keys:
            xIN = key["X"]
            yIN = key["Y"]
            yIN = yIN[:][:].flatten()
            xOUT = xIN[:7]
            xOUT = np.divide(xOUT,x_max)

            #yOUT = yIN[:,y_mask]
            yOUT = np.compress(y_mask,yIN)
            yOUT = np.divide(yOUT,yCOMPMAX)
            group = fileOUT.create_group(key)
            x = group.create_dataset("X",data=xOUT)
            y = group.create_dataset("Y",data=yOUT)
            