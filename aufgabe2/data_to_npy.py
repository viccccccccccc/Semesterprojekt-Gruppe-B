import h5py
import numpy as np


datensatz = h5py.File("data.h5")
key_list = list(datensatz.keys())
X = np.zeros((len(key_list), 8))
Y = np.zeros((len(key_list), 256, 256))
print("init complete")
for key in key_list:
    element = datensatz[key]
    X[int(key)][:7] = element["X"][:7]
    Y[int(key)][:][:] = element["Y"][:][:]
print("init complete")
print(X)
print(type(X[0][0]))
print(Y)
print(type(Y[0][0][0]))

# Teilen Sie die Daten in Trainings- und Testdatensätze auf
np.save("X.npy",X)
np.save("Y.npy",Y)


print("files saved")