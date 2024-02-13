
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib as jl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import h5py
import matplotlib
import optuna






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

model = torch.load("../../../../../../../../../../vol/tmp/gruppe_b/64x64_neu/model_best.tar", map_location=torch.device('cpu'))
model = model.cpu()
pca = jl.load("../../../../../../../../../../vol/tmp/gruppe_b/64x64_neu/pca256_64x64_w0.pkl")
scaler_y = jl.load("../../../../../../../../../../vol/tmp/gruppe_b/64x64_neu/scaler_yPCAneu.joblib")
scaler_x = jl.load("../../../../../../../../../../vol/tmp/gruppe_b/64x64_neu/scaler_xPCAneu.joblib")

def generate(input):#input: 1 dimensionales np array der laenge 7
    for i in range(len(input)):
        input[i] = scaler_x[i].transform(input[i].reshape(-1, 1)).flatten()

    torch_data= torch.from_numpy(input)
    with torch.no_grad():
        torch_data = torch_data.float()
        torch_data = torch_data.cpu()
        outputs = model(torch_data)
        outputs = outputs.numpy()

        outputs = scaler_y.inverse_transform(outputs.reshape(-1, 1)).flatten()

        result = pca.inverse_transform(outputs)
        result = result-1
        result[result<0.2]=0
        result = result.reshape(64,64)
        test= np.max(result)-np.min(result)
        if(test<200):
            result = np.zeros((64,64))
        return result




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
    plt.savefig("../plots/final_cluster_grid_5000.png")




# Definieren Sie die Optimierungsfunktion
def optimize(trial):
    """if trial.number == 0:
        param1 = 1.0980315e+02
        param2 = 1.8954570e+00
        param3 = 6.0119409e+02
        param4 = 5.0033993e+01
        param5 = 2.6013130e+01
        param6 = 3000
        param7 = 1.1081532e+00
    else:"""
    param1 = trial.suggest_float('param1', 70, 150)
    param2 = trial.suggest_float('param2', 1.5, 3.0)
    param3 = trial.suggest_float('param3', 300, 800)
    param4 = trial.suggest_float('param4', 0.0, 100.0)
    param5 = trial.suggest_float('param5', 0.0, 100.0)
    param6 = trial.suggest_categorical('param6', [3000, 5000, 10000])
    param7 = trial.suggest_float('param7', 0.5, param2)



    params = np.array([param1, param2, param3, param4, param5, param6, param7])
    return score(generate(params),False)

# Erstellen Sie einen Studien-Objekt und führen Sie die Optimierung durch
param1_values = np.linspace(70, 150, 20)
param2_values = np.linspace(1.5, 3.0, 20)
param3_values = np.linspace(300, 800, 20)
param4_values = np.linspace(0.0, 100.0, 20)
param5_values = np.linspace(0.0, 100.0, 20)
param6_values = [3000, 5000, 10000]  # Since this is categorical, we'll keep the original values
param7_values = np.linspace(0.5, 3.0, 20)  # param7 depends on param2, so we'll generate it dynamically

search_space = {
    'param1': list(param1_values),
    'param2': list(param2_values),
    'param3': list(param3_values),
    'param4': list(param4_values),
    'param5': list(param5_values),
    'param6': param6_values,
    'param7': list(param7_values)
}
study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space=search_space),direction='maximize')
study.optimize(optimize, n_trials=5000)

# Drucken Sie das beste Ergebnis
print(study.best_params)



result = generate(np.array(list(study.best_params.values())))
score(result,True)
result = result.reshape(64,64)

plt.figure(0)
plt.imshow(result,cmap='turbo')
plt.show()




