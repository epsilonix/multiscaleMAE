import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
from joblib import Parallel, delayed

def load_embeddings(file_path):
    return np.load(file_path)

def calculate_kmeans_and_scores(n_clusters, embeddings):
    try:
        print(f"Calculating for {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(embeddings)
        distortion = kmeans.inertia_
        silhouette_avg = silhouette_score(embeddings, kmeans.labels_)
        return n_clusters, distortion, silhouette_avg
    except Exception as e:
        print(f"Error calculating for {n_clusters} clusters: {e}")
        return n_clusters, None, None

def plot_elbow_method(k_values, distortions, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, distortions, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Distortion')
    plt.title('Elbow Method For Optimal k')
    plt.savefig(os.path.join(output_path, 'elbow_plot.png'))
    plt.close()

def plot_silhouette_scores(k_values, silhouette_scores, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, silhouette_scores, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores For Optimal k')
    plt.savefig(os.path.join(output_path, 'silhouette_plot.png'))
    plt.close()

def main():
    input_path = '/gpfs/scratch/ss14424/Brain/channels_20/cells/analysis_output_all/tile_embedding/embedding_mean.npy'
    output_path = '/gpfs/scratch/ss14424/Brain/channels_20/cells/analysis_output_all/tile_embedding/'

    embeddings = load_embeddings(input_path)

    k_values = range(5, 51)

    # Parallel processing for KMeans and silhouette score computation
    results = Parallel(n_jobs=-1)(delayed(calculate_kmeans_and_scores)(k, embeddings) for k in k_values)

    # Filter out results with errors
    k_values, distortions, silhouette_scores = zip(*[(k, d, s) for k, d, s in results if d is not None and s is not None])

    plot_elbow_method(k_values, distortions, output_path)
    plot_silhouette_scores(k_values, silhouette_scores, output_path)

    print("Elbow plot and silhouette plot have been saved.")

if __name__ == "__main__":
    main()
