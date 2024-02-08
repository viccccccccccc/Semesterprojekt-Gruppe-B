import h5py
import joblib as jl
import numpy as np


scaler_y = jl.load("scaler_yDECON2.joblib")
y=np.full((4096),1)
y = scaler_y.inverse_transform(y.reshape(-1,1)).flatten()
print(y[1])


"""data = "data2m.h5"
min = [10000000,10000000,10000000,10000000,10000000,10000000,10000000]
max = [0,0,0,0,0,0,0]
with h5py.File(data, "r") as file:
    sample = "252"

    for i in range(1000000):
        img = np.array(file[str(i)]["X"][:])
        for j in range(7):
            if img[j] < min[j]:
                min[j]=img[j]
            if img[j] > max[j]:
                max[j]=img[j]
        if i%100==0:
            print("after",i)
            for j in range(7):
                print(f'{j}, has min {min[j]} and max {max[j]}')
"""

 #min_max=[[70,1.5,300,0,0,3000,0.5],[150,3.0,800,100,100,10000,3]]