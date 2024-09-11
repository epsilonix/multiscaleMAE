import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from joblib import Parallel, delayed
import argparse

def clustering(emb_path, n_clusters, save_path):

    embedding = np.load(emb_path)
    print(f'Embedding loaded with path {emb_path}')
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embedding)
    kmeans_labels = kmeans.labels_
    print(f'kmeans_labels: {kmeans_labels}')
    kmeans_inertia = kmeans.inertia_
    
    # Create the "clustering" directory within save_path
    clustering_dir = os.path.join(os.path.dirname(save_path), 'clustering')
    os.makedirs(clustering_dir, exist_ok=True)

    # Save the KMeans labels
    labels_save_path = os.path.join(clustering_dir, 'labels_{n_clusters}.npy')
    np.save(labels_save_path, kmeans_labels)
    print(f'Saved K-means labels to {labels_save_path}')
    
    # Save KMeans inertia
    inertia_save_path = os.path.join(clustering_dir, 'kmeans_inertia.txt')
    with open(inertia_save_path, 'w') as f:
        f.write(str(kmeans_inertia))
    print(f'K-means inertia saved to {inertia_save_path}')
    
#    # No subsampling - use the entire dataset
#    embedding_sample = embedding
#    
#    # Range of clusters to try - every 5th cluster from 5 to 70
#    k_range = range(5, 71, 5)
#
#    # Parallelize silhouette score computation on the entire dataset
#    silhouette_scores = Parallel(n_jobs=-1)(
#        delayed(silhouette_score)(
#            embedding_sample, 
#            KMeans(n_clusters=k, random_state=42).fit(embedding_sample).labels_
#        ) 
#        for k in k_range
#    )
#
#    # Plotting the Elbow Method graph
#    plt.figure(figsize=(10, 5))
#    plt.plot(k_range, [KMeans(n_clusters=k, random_state=42).fit(embedding_sample).inertia_ for k in k_range], '-o')
#    plt.title('Elbow Method For Optimal k')
#    plt.xlabel('Number of clusters, k')
#    plt.ylabel('Inertia')
#    plt.xticks(k_range)
#    elbow_plot_path = os.path.join(clustering_dir, 'elbow_method_plot.png')
#    plt.savefig(elbow_plot_path)  # Save the figure
#    plt.close()  # Close the plot to free memory
#    print(f'Elbow Method plot saved at {elbow_plot_path}')
#
#    # Plotting the Silhouette Scores
#    plt.figure(figsize=(10, 5))
#    plt.plot(k_range, silhouette_scores, '-o')
#    plt.title('Silhouette Score For Each k')
#    plt.xlabel('Number of clusters, k')
#    plt.ylabel('Silhouette Score')
#    plt.xticks(k_range)
#    silhouette_plot_path = os.path.join(clustering_dir, 'silhouette_score_plot.png')
#    plt.savefig(silhouette_plot_path)  # Save the figure
#    plt.close()  # Close the plot to free memory
#    print(f'Silhouette Score plot saved at {silhouette_plot_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="KMeans clustering")
    parser.add_argument('emb_path', type=str, help='Path to the embedding file (npy format)')
    parser.add_argument('n_clusters', type=int, help='Number of clusters for KMeans')
    parser.add_argument('save_path', type=str, help='Path to save the KMeans labels (npy format)')
    
    args = parser.parse_args()
    
    clustering(args.emb_path, args.n_clusters, args.save_path)
