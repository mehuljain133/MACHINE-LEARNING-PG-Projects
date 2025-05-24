# Unit-V Unsupervised learning: Clustering, distance metrics, Mixture models, Expectation Maximization, Cluster validation methods

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import cdist, cosine

# Generate synthetic dataset with 3 clusters
X, y_true = make_blobs(n_samples=500, centers=3, cluster_std=0.60, random_state=0)

# Visualize original data
plt.scatter(X[:, 0], X[:, 1], s=30)
plt.title("Original Data")
plt.grid()
plt.show()

# --------- K-Means Clustering ---------
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X)
print("K-Means Cluster Centers:\n", kmeans.cluster_centers_)

plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', s=30)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=200, c='red', marker='X')
plt.title("K-Means Clustering")
plt.grid()
plt.show()

# --------- Hierarchical Clustering ---------
agglo = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
agglo_labels = agglo.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=agglo_labels, cmap='rainbow', s=30)
plt.title("Agglomerative Hierarchical Clustering")
plt.grid()
plt.show()

# --------- Distance Metrics ---------
# Compute Euclidean distances between first 5 points and cluster centers (KMeans)
euclid_distances = cdist(X[:5], kmeans.cluster_centers_, metric='euclidean')
print("Euclidean distances (first 5 samples to cluster centers):\n", euclid_distances)

# Compute Cosine distances (same points)
cosine_distances = cdist(X[:5], kmeans.cluster_centers_, metric='cosine')
print("Cosine distances (first 5 samples to cluster centers):\n", cosine_distances)

# --------- Gaussian Mixture Model (GMM) with EM Algorithm ---------
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm_labels = gmm.fit_predict(X)
print("GMM Means:\n", gmm.means_)

plt.scatter(X[:, 0], X[:, 1], c=gmm_labels, cmap='coolwarm', s=30)
plt.title("Gaussian Mixture Model Clustering")
plt.grid()
plt.show()

# --------- Cluster Validation Metrics ---------
print("K-Means Silhouette Score:", silhouette_score(X, kmeans_labels))
print("Agglomerative Silhouette Score:", silhouette_score(X, agglo_labels))
print("GMM Silhouette Score:", silhouette_score(X, gmm_labels))

print("K-Means Davies-Bouldin Index:", davies_bouldin_score(X, kmeans_labels))
print("Agglomerative Davies-Bouldin Index:", davies_bouldin_score(X, agglo_labels))
print("GMM Davies-Bouldin Index:", davies_bouldin_score(X, gmm_labels))
