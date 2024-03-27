import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
#sys.path.append('/gpfs/data/tsirigoslab/home/jt3545/plt-figure')
#from core import subplots

def main():
    umap_emb_path = sys.argv[1]
    marker_path = sys.argv[2]
    plot_path = sys.argv[3]
    #plot_umap_markers_separate(umap_emb_path, marker_path, plot_path)
    plot(umap_emb_path, marker_path, plot_path)

def plot(umap_path, marker_path, plot_path, channel_names = None):
    umap_embedding = np.load(umap_path)
    markers = np.load(marker_path)

    cols = 5
    rows = len(channel_names) // cols + 1
    #fig, axes = subplots(rows, cols, figsize=(cols * 50, rows * 40), font_size = 5)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 50, rows * 40))

    for marker_i in range(len(channel_names)):
        marker_color = markers[:, marker_i]
        
        m_min, m_max = np.percentile(marker_color, [10, 90])
        marker_color = np.clip((marker_color - m_min) / m_max, 0, 1)
        
        sc = axes[marker_i // cols, marker_i % cols].scatter(umap_embedding[:, 0], umap_embedding[:, 1], s = 0.3, c = marker_color, cmap = 'viridis')
        axes[marker_i // cols, marker_i % cols].set_title(f'Marker: {channel_names[marker_i]}')
        fig.colorbar(sc, ax=axes[marker_i // cols, marker_i % cols])

        # Dark background
        #axes[marker_i // cols, marker_i % cols].set_facecolor('black')

    # Clean up empty axes
    for i in range(len(channel_names), rows * cols):
        axes[i // cols, i % cols].axis('off')

    for ax in axes.flat:
        ax.label_outer()
    plt.savefig(plot_path, bbox_inches='tight')
    plt.clf()

def plot_umap_markers_separate(umap_path, marker_path, plot_path):
    umap_embedding = np.load(umap_path)
    markers = np.load(marker_path)

    channel_names = ["CD117", "CD11c", "CD14", "CD163", "CD16", "CD20", "CD31", "CD3", "CD4", "CD68", "CD8a", "CD94", "DNA1", "FoxP3", "HLA-DR", "MPO", "Pancytokeratin", "TTF1"]

    for marker_i in range(len(channel_names)):
        marker_color = markers[:, marker_i]
        
        m_min, m_max = np.percentile(marker_color, [5, 95])
        marker_color = np.clip((marker_color - m_min) / m_max, 0, 1)

        fig, ax = subplots(figsize=(20, 17), font_size = 5)

        if channel_names[marker_i] in ['CD3', 'CD4', 'CD8a', 'FoxP3']:
            cmap = 'Reds'
        elif channel_names[marker_i] in ['CD68', 'CD163']:
            cmap = 'Oranges'
        elif channel_names[marker_i] in ['Pancytokeratin', 'TTF1']:
            cmap = 'Purples'
        elif channel_names[marker_i] == 'CD20':
            cmap = 'Greens'
        elif channel_names[marker_i] in ['MPO', 'CD14', 'CD16']:
            cmap = 'Blues'
        else:
            cmap = 'Greys'
        
        sc = ax.scatter(umap_embedding[:, 0], umap_embedding[:, 1], s = 0.1, c = marker_color, cmap = cmap, edgecolors = 'none')
        ax.set_title(f'{channel_names[marker_i]}')
        # Remove all frame
        ax.axis('off')

        fig.colorbar(sc, ax=ax)

        plot_path_marker = plot_path.replace('.pdf', f'_{channel_names[marker_i]}.pdf')
        plot_path_marker = plot_path.replace('.png', f'_{channel_names[marker_i]}.png')

        plt.savefig(plot_path_marker, bbox_inches='tight', dpi=900)
        plt.clf()


if __name__ == '__main__':
    main()

