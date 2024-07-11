import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import sklearn.cluster as cluster

def main():
    emb_path = sys.argv[1]
    umap_path = sys.argv[2]
    n_clusters = int(sys.argv[3])
    save_path = sys.argv[4]

    clustering(emb_path, umap_path, n_clusters, save_path)

def clustering(emb_path, umap_path, n_clusters, save_path):

    embedding = np.load(emb_path)
    print(f'Embedding loaded from path {emb_path}')
    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=0).fit(embedding)
    kmeans_labels = kmeans.labels_
    print(f'kmeans_labels:{kmeans_labels}')
    kmeans_inertia = kmeans.inertia_
    
    np.save(save_path, kmeans_labels)
    print(f'K-means labels saved to {save_path}')
    
    with open(save_path.replace('.npy', '_inertia.txt'), 'w') as f:
        f.write(str(kmeans_inertia))
    print('Kmeans inertia saved')

#    print('Plotting clusters on UMAP')
#    umap_embedding = np.load(umap_path)
#    fig, ax = plt.subplots(figsize=(10, 10))
#    if n_clusters <= 20:
#        palette = 'tab20'
#    else:
#        palette = 'gist_ncar'
#        
#    sns.scatterplot(x=umap_embedding[:, 0], y=umap_embedding[:, 1], size=1, hue=kmeans_labels, palette=palette, legend='full')
#    plt.title(f'K-means {n_clusters} clusters')
#    plt.legend(bbox_to_anchor=(1.1, 1.05))
#    plt.savefig(save_path.replace('.npy', '.png'))
#    plt.clf()
    
    # Range of clusters to try
    k_range = range(2, 50)

    inertias = []  # List to collect the within-cluster sum of squares
    silhouette_scores = []  # List to collect the silhouette scores

    for k in k_range:
        kmeans = cluster.KMeans(n_clusters=k, random_state=42).fit(embedding)
        inertias.append(kmeans.inertia_)
        score = silhouette_score(embedding, kmeans.labels_)
        silhouette_scores.append(score)

    # Save the plots in the same directory as save_path
    plot_save_dir = os.path.dirname(save_path)
    
    # Plotting the Elbow Method graph
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, inertias, '-o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters, k')
    plt.ylabel('Inertia')
    plt.xticks(k_range)
    elbow_plot_path = os.path.join(plot_save_dir, 'elbow_method_plot.png')
    plt.savefig(elbow_plot_path)  # Save the figure
    plt.close()  # Close the plot to free memory

    # Optional: Plot Silhouette Scores to compare
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, silhouette_scores, '-o')
    plt.title('Silhouette Score For Each k')
    plt.xlabel('Number of clusters, k')
    plt.ylabel('Silhouette Score')
    plt.xticks(k_range[1:])
    silhouette_plot_path = os.path.join(plot_save_dir, 'silhouette_score_plot.png')
    plt.savefig(silhouette_plot_path)  # Save the figure
    plt.close()  # Close the plot to free memory

if __name__ == '__main__':
    main()
