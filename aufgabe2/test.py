import h5py

datensatz = h5py.File("../../../../../../../../../vol/tmp/gruppe_b/pca256.h5")
key_list = list(datensatz.keys())
print(len(key_list))