import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib
matplotlib.use("Agg")

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
data = "../../A2/rzp-1_sphere1mm_train_2million_bin32.h5"
with h5py.File(data, "r") as file:
    sample = "252"
    img = np.array(file[sample]["Y"][:])

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


#create plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 12))

ax1.imshow(original_img, cmap='turbo', aspect='auto')
ax1.set_title(f"Sample: {sample}  Score: {silhouette_avg:.6f}")

colors = ['red', 'blue', 'green']
for i, cluster in enumerate(clusters):
    y, x = zip(*cluster)
    ax2.scatter(x, y, c=colors[i], label=f'Cluster {i+1}', marker="x", alpha=0.6, s=20)

ax2.set_xlim(0, img.shape[1])
ax2.set_ylim(img.shape[0], 0)

plt.tight_layout()
plt.savefig(f"../../A2/plots/final_cluster_{sample}.png")
