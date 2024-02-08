import numpy as np
import h5py
from torch.utils.data import Dataset
from sklearn.decomposition import IncrementalPCA
import joblib
import random
import matplotlib.pyplot as plt

old_file = h5py.File("../../../../../../../../../vol/tmp/feuforsp/rzp-1_sphere1mm_train_2million_bin32.h5", "r")
#pca_file = h5py.File("pca256.h5", "r")
key_list = list(old_file.keys())

pca = joblib.load("pcaneu/pca256_64x64_w0.pkl")

while True:
    key = random.choice(key_list)
    old_element = old_file[key]
    #pca_element = pca_file[key]
    d1 = old_element["Y"][:][:]
    pca_element = pca.transform([d1.flatten()])
    d2 = pca_element.flatten()
    #xo = old_element["X"][:] 
    #xp = pca_element["X"][:]
    d3 = pca.inverse_transform(d2)
    d3 = d3-1
    d3[d3<0.2]=0
    #if not np.any(d1 != 0):
    d3 = d3.reshape(64,64)
    #print("Xs:")
    #print(str(xo))
    #print(str(xp))

    pf, axarr = plt.subplots(1,2)
    axarr[0].imshow(d1, cmap='turbo')  # Verwenden Sie 'viridis' Farbschema für das erste Bild
    axarr[1].imshow(d3, cmap='turbo')  # Verwenden Sie 'plasma' Farbschema für das zweite Bild

    plt.show()