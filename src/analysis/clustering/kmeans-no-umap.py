import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import sklearn.cluster as cluster

def main():
    emb_path = '/gpfs/scratch/ss14424/Brain/cells/analysis_output/tile_embedding/embedding_mean.npy'
    n_clusters = 20
    save_path = '/gpfs/scratch/ss14424/Brain/cells/analysis_output/labels_20'

    clustering(emb_path, n_clusters, save_path)

def clustering(emb_path, n_clusters, save_path):

    embedding = np.load(emb_path)
    print(f'Embedding loaded with path {embedding}')
    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=0).fit(embedding)
    kmeans_labels = kmeans.labels_
    print(f'kmeans_labels:{kmeans.labels_}')
    kmeans_inertia = kmeans.inertia_
    
    
    np.save(save_path, kmeans_labels)
    print(f'emb_path is {emb_path} and K-mean label:{kmeans_labels}')
    with open(save_path.split('.')[0] + '_inertia.txt', 'w') as f:
        f.write(str(kmeans_inertia))
    print('Kmeans inertia saved')
    
    # Range of clusters to try
    k_range = range(2, 60)

    inertias = [] # List to collect the within-cluster sum of squares
    silhouette_scores = [] # List to collect the silhouette scores

    for k in k_range:
        kmeans = cluster.KMeans(n_clusters=k, random_state=42).fit(embedding)
        inertias.append(kmeans.inertia_)
        # Compute the silhouette score only if there are 2 or more clusters
        score = silhouette_score(embedding, kmeans.labels_)
        silhouette_scores.append(score)

    # Specify the output directory for the plots
    output_dir = '/gpfs/scratch/ss14424/Brain/plots/'
    os.makedirs(output_dir, exist_ok=True) # Create the output directory if it doesn't exist

    # Plotting the Elbow Method graph
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, inertias, '-o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters, k')
    plt.ylabel('Inertia')
    plt.xticks(k_range)
    elbow_plot_path = os.path.join(output_dir, 'elbow_method_plot.png')
    plt.savefig(elbow_plot_path) # Save the figure
    plt.close() # Close the plot to free memory

    # Optional: Plot Silhouette Scores to compare
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, silhouette_scores, '-o') # Adjusting k_range for silhouette since it starts from 2
    plt.title('Silhouette Score For Each k')
    plt.xlabel('Number of clusters, k')
    plt.ylabel('Silhouette Score')
    plt.xticks(k_range[1:])
    silhouette_plot_path = os.path.join(output_dir, 'silhouette_score_plot.png')
    plt.savefig(silhouette_plot_path) # Save the figure
    plt.close() # Close the plot to free memory

if __name__ == '__main__':
    main()
