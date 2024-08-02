import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from joblib import Parallel, delayed

def clustering(emb_path, n_clusters, save_path):

    embedding = np.load(emb_path)
    print(f'Embedding loaded with path {emb_path}')
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embedding)
    kmeans_labels = kmeans.labels_
    print(f'kmeans_labels: {kmeans_labels}')
    kmeans_inertia = kmeans.inertia_
    
    np.save(os.path.join(save_path, 'kmeans_labels.npy'), kmeans_labels)
    print(f'Saved K-means labels to {os.path.join(save_path, "kmeans_labels.npy")}')
    with open(os.path.join(save_path, 'kmeans_inertia.txt'), 'w') as f:
        f.write(str(kmeans_inertia))
    print('K-means inertia saved')
    
    # Subsampling for silhouette score calculation
    if len(embedding) > 10000:
        embedding_sample = embedding[np.random.choice(embedding.shape[0], 10000, replace=False)]
    else:
        embedding_sample = embedding
    
    # Range of clusters to try - every 5th cluster from 5 to 70
    k_range = range(5, 71, 5)

    # Parallelize silhouette score computation
    silhouette_scores = Parallel(n_jobs=-1)(
        delayed(silhouette_score)(
            embedding_sample, 
            KMeans(n_clusters=k, random_state=42).fit(embedding_sample).labels_
        ) 
        for k in k_range
    )

    # Specify the output directory for the plots
    output_dir = os.path.join(save_path, 'plots')
    os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

    # Plotting the Elbow Method graph
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, [KMeans(n_clusters=k, random_state=42).fit(embedding_sample).inertia_ for k in k_range], '-o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters, k')
    plt.ylabel('Inertia')
    plt.xticks(k_range)
    elbow_plot_path = os.path.join(output_dir, 'elbow_method_plot.png')
    plt.savefig(elbow_plot_path)  # Save the figure
    plt.close()  # Close the plot to free memory

    # Optional: Plot Silhouette Scores to compare
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, silhouette_scores, '-o')
    plt.title('Silhouette Score For Each k')
    plt.xlabel('Number of clusters, k')
    plt.ylabel('Silhouette Score')
    plt.xticks(k_range)
    silhouette_plot_path = os.path.join(output_dir, 'silhouette_score_plot.png')
    plt.savefig(silhouette_plot_path)  # Save the figure
    plt.close()  # Close the plot to free memory

if __name__ == '__main__':
    save_path = '/gpfs/scratch/ss14424/Brain/channels_37/tiles/analysis_output_64/'  # Replace with your actual save path
    emb_path = os.path.join(save_path, 'tile_embedding', 'embedding_mean.npy')
    n_clusters = 40  # Specify the number of clusters for the initial KMeans clustering
    clustering(emb_path, n_clusters, save_path)
