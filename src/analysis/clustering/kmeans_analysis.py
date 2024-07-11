import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import os
import time
from joblib import Parallel, delayed

def load_embeddings(file_path):
    return np.load(file_path)

def calculate_kmeans_and_scores(n_clusters, embeddings):
    try:
        print(f"Calculating for {n_clusters} clusters...")
        start_time = time.time()
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=5, batch_size=1000).fit(embeddings)
        distortion = kmeans.inertia_
        silhouette_avg = silhouette_score(embeddings, kmeans.labels_)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Finished calculating for {n_clusters} clusters in {elapsed_time:.2f} seconds.")
        return n_clusters, distortion, silhouette_avg, elapsed_time
    except Exception as e:
        print(f"Error calculating for {n_clusters} clusters: {e}")
        return n_clusters, None, None, 0

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

    k_values = range(5, 31)
    distortions = []
    silhouette_scores = []
    elapsed_times = []

    total_start_time = time.time()

    # Parallel processing with controlled memory usage
    results = Parallel(n_jobs=4)(delayed(calculate_kmeans_and_scores)(k, embeddings) for k in k_values)

    for result in results:
        n_clusters, distortion, silhouette_avg, elapsed_time = result
        if distortion is not None and silhouette_avg is not None:
            distortions.append(distortion)
            silhouette_scores.append(silhouette_avg)
            elapsed_times.append(elapsed_time)

    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time

    plot_elbow_method(k_values, distortions, output_path)
    plot_silhouette_scores(k_values, silhouette_scores, output_path)

    print(f"Total elapsed time for all calculations: {total_elapsed_time:.2f} seconds.")
    print("Elbow plot and silhouette plot have been saved.")

if __name__ == "__main__":
    main()
