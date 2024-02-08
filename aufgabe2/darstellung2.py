import datetime
import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib as jl
import torch
import h5py
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import h5py
import matplotlib.pyplot as plt



def find_max_intensity(img):
    return np.unravel_index(np.argmax(img), img.shape)

def find_cluster(img, max_point, threshold_fraction):
    cluster_points = [max_point]
    points_to_check = [max_point]
    max_value = img[max_point]

    threshold = max_value * threshold_fraction

    while points_to_check:
        new_points = []
        for point in points_to_check:
            y, x = point
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < img.shape[0] and 0 <= nx < img.shape[1] and
                        img[ny, nx] > threshold and (ny, nx) not in cluster_points):
                        new_points.append((ny, nx))
                        cluster_points.append((ny, nx))
        points_to_check = new_points

    return cluster_points

def average_distance(point, points):
    return np.mean([np.linalg.norm(np.array(point) - np.array(other_point)) for other_point in points])

def silhouette_score(clusters, img):
    
    cluster_indices = [np.ravel_multi_index(np.array(cluster).T, img.shape) for cluster in clusters]

    all_scores = []
    for cluster_idx, cluster in enumerate(cluster_indices):
        for point in cluster:
            a = average_distance(point, [p for p in cluster if p != point])
            other_clusters = [c for i, c in enumerate(cluster_indices) if i != cluster_idx]
            b = min(average_distance(point, other_cluster) for other_cluster in other_clusters)
            score = abs(b - a) / max(a, b)  # Use absolute value to ensure score is positive
            all_scores.append(score)

    silhouette_avg = np.mean(all_scores)
    return silhouette_avg

# Load data
def score(imgo,print):
    img=imgo.copy()
    original_img = img.copy()

    # Initialize clusters and thresholds
    clusters = []
    threshold_fractions = [0.09, 0.25, 0.001]


    # Clustering process
    for i in range(3):
        max_point = find_max_intensity(img)
        if img[max_point] == 0:
            break

        cluster = find_cluster(img, max_point, threshold_fractions[i])
        clusters.append(cluster)
        for point in cluster:
            img[point] = 0  # Set the cluster points to 0

    # After clustering, calculate the average silhouette score
    silhouette_avg = silhouette_score(clusters, img)

    if not print:
        return silhouette_avg
    #create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 12))

    ax1.imshow(original_img, cmap='turbo', aspect='auto')
    ax1.set_title(f"Sample: gen  Score: {silhouette_avg:.6f}")

    colors = ['red', 'blue', 'green']
    for i, cluster in enumerate(clusters):
        y, x = zip(*cluster)
        ax2.scatter(x, y, c=colors[i], label=f'Cluster {i+1}', marker="x", alpha=0.6, s=20)

    ax2.set_xlim(0, img.shape[1])
    ax2.set_ylim(img.shape[0], 0)

    plt.tight_layout()
    #plt.savefig(f"../../A2/plots/final_cluster_gen.png")



class CustomDataset(Dataset):
    def __init__(self, hdf5_path):
        self.x_len = 7
        self.y_len = 256 * 256
        self.file = h5py.File(hdf5_path, "r")
        self.key_list = list(self.file.keys())
        self.scaler_x = [MinMaxScaler() for _ in range(self.x_len)]
        self.scaler_y = MinMaxScaler()
        self.normalized = False

    def normalize(self):
        if os.path.exists("scaler_xDECON2.joblib") and os.path.exists("scaler_yDECON2.joblib"):
            self.scaler_x = jl.load("scaler_xDECON2.joblib")
            self.scaler_y = jl.load("scaler_yDECON2.joblib")
        else:
            for idx in range(len(self.key_list)):
                datapoint = self.file[self.key_list[idx]]
                x = datapoint["X"][:self.x_len]
                y = (datapoint["Y"][:][:]).flatten()

                # Fit all scales with x and y
                for i in range(self.x_len):
                    self.scaler_x[i].partial_fit(x[i].reshape(-1, 1))
                self.scaler_y.partial_fit(y.reshape(-1, 1))

            jl.dump(self.scaler_x, "scaler_xDECON2.joblib")
            jl.dump(self.scaler_y, "scaler_yDECON2.joblib")

        self.normalized = True

    def denormalize_y(self, y):
        if self.normalized:
            for i in range(self.x_len):
                y[i] = self.scaler_x[i].inverse_transform(y[i].reshape(-1, 1)).flatten()
        return y

    def __len__(self):
        return len(self.key_list)

    def __getitem__(self, idx):
        if isinstance(idx, str):
            datapoint = self.file[idx]
        else:
            datapoint = self.file[self.key_list[idx]]

        x = datapoint["X"][:self.x_len]
        y = (datapoint["Y"][:][:]).flatten()

        # Normalize each group in x and y
        if self.normalized:
            for i in range(self.x_len):
                x[i] = self.scaler_x[i].transform(x[i].reshape(-1, 1)).flatten()
            #y = self.scaler_y.transform(y.reshape(-1,1)).flatten()

        return x, y
    
    def getX(self,idx):
        if isinstance(idx, str):
            datapoint = self.file[idx]
        else:
            datapoint = self.file[self.key_list[idx]]

        x = datapoint["X"][:self.x_len]
        return x

    def split(self, anteil_test):
        keys = self.key_list.copy()
        random.shuffle(keys)
        split_index = int(len(keys) * anteil_test)
        test_keys = keys[:split_index]
        train_keys = keys[split_index:]

        return CustomView(self, train_keys), CustomView(self, test_keys)



class CustomView(Dataset):
    def __init__(self, my_reader, key_list):
        self.reader = my_reader
        self.key_list = key_list

    def __len__(self):
        return len(self.key_list)

    def __getitem__(self, idx):
        return self.reader[self.key_list[idx]]

class ParameterToImage(nn.Module):
    def __init__(self):
        super(ParameterToImage, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(7, 256),
            nn.ReLU()
        )

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(7, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 256),
    
        )

    def forward(self, x):
        return self.fc(x)


data = CustomDataset("../../../../../../../../../vol/tmp/feuforsp/rzp-1_sphere1mm_train_2million_bin32.h5")
data.normalize()
dl = DataLoader(data, batch_size=1,  num_workers=6, shuffle=True)
model = torch.load("30.01.24, 11:07:54_pca_no_sced/model_best.tar", map_location=torch.device('cpu'))
model = model.cpu()
pca = jl.load("../../../../../../../../../vol/tmp/gruppe_b/64x64/pca256.pkl")
scaler_y = jl.load("scaler_yPCA.joblib")
filter = np.load("nullBild.npz")['name1']

trueNegative=0
truePositive=0
falseNegative=0
falsePositive=0

for i in range(len(dl)):
    inputs, d1 = next(iter(dl))
    dnp = d1.numpy()
    #if not np.any(dnp != 0):
        

    with torch.no_grad():
        inputs = inputs.float()
        inputs = inputs.cpu()
        outputs = model(inputs)
        outputs = outputs.numpy()
        #print(outputs.shape)
        outputs = scaler_y.inverse_transform(outputs.reshape(-1, 1)).flatten()
        #print(outputs.shape)
        d2=pca.inverse_transform(outputs)
    
    

    #np.savez('nullBild.npz',name1=d1)
    d1 = d1.reshape(64,64)
    d2 = d2.reshape(64,64)

    #test = np.sum(abs(d2 - filter))
    test= np.max(d2)-np.min(d2)
    test = np.sum(d2[d2<0])

    """if(test<200):
        if not np.any(dnp != 0):
            truePositive+=1
        else:
            falsePositive+=1"""
    

    score = score(d1.numpy(),False)
    if(score>0.69):
        print(score)
        print(data.denormalize_y(inputs.numpy().reshape(-1, 1)))
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(d1)
        axarr[1].imshow(d2)
        plt.show()
    """else:
        if not np.any(dnp != 0):
            falseNegative+=1
            print(test)
        else:
            trueNegative+=1"""

#print(f'truePositive: {truePositive}, falsePositive: {falsePositive}, trueNegative: {trueNegative}, falseNegative: {falseNegative}')

