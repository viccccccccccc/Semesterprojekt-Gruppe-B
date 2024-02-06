
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib as jl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt



model = torch.load("30.01.24, 11:07:54_pca_no_sced/model_best.tar", map_location=torch.device('cpu'))
model = model.cpu()
pca = jl.load("../../../../../../../../../vol/tmp/gruppe_b/pca256.pkl")
scaler_y = jl.load("scaler_yPCA.joblib")



def generate(input):#input: 1 dimensionales np array der laenge 7
    torch_data= torch.form_numpy(input)
    with torch.no_grad():
        torch_data = torch_data.float()
        torch_data = torch_data.cpu()
        outputs = model(torch_data)
        outputs = outputs.numpy()

        outputs = scaler_y.inverse_transform(outputs.reshape(-1, 1)).flatten()

        result = pca.inverse_transform(outputs)
        result[result<0]=0
        result = result.reshape(64,64)
        test= np.max(result)-np.min(result)
        if(test<200):
            result = np.zeros((64,64))
        return result
