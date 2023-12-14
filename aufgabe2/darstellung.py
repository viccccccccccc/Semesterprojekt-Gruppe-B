import random
import h5py
import matplotlib.pyplot as plt
f = h5py.File("data.h5")
keys = list(f.keys())

for i in range(30):
    element = f[random.choice(keys)]
    d1 = element["X"]
    d2 = element["Y"]
    plt.figure(i)
    plt.imshow(d2)
plt.show()




