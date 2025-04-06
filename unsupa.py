import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
data = pd.read_csv('data.csv')  # Replace with your unsupervised dataset path

# 2. Drop non-numeric / ID columns if needed
if 'id' in data.columns:
    data.drop(columns=['id'], inplace=True)

# 3. Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(data)

# 4. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# 5. PCA for dimensionality reduction
pca = PCA(n_components=0.95)  # Keep 95% variance
X_pca = pca.fit_transform(X_scaled)

# 6. Try different cluster sizes and store silhouette scores
sil_scores = []
n_components_range = range(2, 11)  # Trying 2 to 10 clusters

for n in n_components_range:
    gmm = GaussianMixture(n_components=n, random_state=42)
    labels = gmm.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    sil_scores.append(score)
    print(f"Clusters: {n}, Silhouette Score: {score:.4f}")

# 7. Plot Silhouette Scores
plt.figure(figsize=(8, 5))
sns.lineplot(x=n_components_range, y=sil_scores, marker='o')
plt.title("Silhouette Score vs Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()

# 8. Choose best number of clusters
best_k = n_components_range[np.argmax(sil_scores)]
print(f"\nBest number of clusters based on Silhouette Score: {best_k}")

# 9. Fit GMM with optimal clusters
final_gmm = GaussianMixture(n_components=best_k, random_state=42)
final_labels = final_gmm.fit_predict(X_pca)

# 10. Save clustering result
clustered_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
clustered_df['Cluster'] = final_labels
clustered_df.to_csv("clustered_output.csv", index=False)
print("Clustered data saved to clustered_output.csv")


# KMEANS

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
data = pd.read_csv('data.csv')  # Replace with your dataset path

# 2. Drop non-numeric / ID columns
if 'id' in data.columns:
    data.drop(columns=['id'], inplace=True)

# 3. Fill missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(data)

# 4. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# 5. Dimensionality reduction (optional but useful for clustering)
pca = PCA(n_components=0.95)  # Retain 95% variance
X_pca = pca.fit_transform(X_scaled)

# 6. Find best number of clusters using Silhouette Score
sil_scores = []
inertias = []
k_range = range(2, 11)  # Try 2 to 10 clusters

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    sil_scores.append(score)
    inertias.append(kmeans.inertia_)
    print(f"K: {k}, Silhouette Score: {score:.4f}, Inertia: {kmeans.inertia_:.2f}")

# 7. Plot Silhouette Score vs K
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.lineplot(x=k_range, y=sil_scores, marker='o', label='Silhouette Score')
plt.title('Silhouette Score vs Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid(True)

# 8. Also plot Inertia (Elbow Method)
plt.subplot(1, 2, 2)
sns.lineplot(x=k_range, y=inertias, marker='o', color='orange', label='Inertia')
plt.title('Elbow Curve (Inertia vs K)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)

plt.tight_layout()
plt.show()

# 9. Pick optimal k based on Silhouette Score or Elbow point
best_k = k_range[np.argmax(sil_scores)]
print(f"\nBest number of clusters based on Silhouette Score: {best_k}")

# 10. Final model
final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
final_labels = final_kmeans.fit_predict(X_pca)

# 11. Save results
clustered_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
clustered_df['Cluster'] = final_labels
clustered_df.to_csv('kmeans_clustered_output.csv', index=False)
print("Clustered data saved to kmeans_clustered_output.csv")
