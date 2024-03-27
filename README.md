# CANVAS-Dev
Automatic analysis of multiplexed images

This is a python project that analyze multiplexed immunoflorecent images using deep learning models.
It can be splitted into several parts.

I want you to generate prompt for specific parts and include input and output specifications.

1. The first part is converting a qptiff file into zarr arrays.
2. The second part is preprocess that image and tile it into different regions of certain size.
3. The third part is build a pytorch dataloader that loads this zarr file. I already have a file that you can adapt to.
4. Then we need to train a masked autoencoder model on the tiles.
5. Then we need to use the trained encoder to encode all tiles and save the results.
6. Then we will run a kmeans clustering on the tiles and return the cluster number for each tile.
7. Then we map the color of the cluter back on to the original slide for visualizaiton.

tanjimin: Help me generate specifications for each step and append them to the end of this file

GitHub Copilot: Sure, here are the specifications for each step:

1. **Convert qptiff file to zarr arrays**
   - Input: path to qptiff file
   - Output: zarr arrays saved to disk
2. **Preprocess and tile image**
   - Input: path to zarr arrays, tile size, overlap size
   - Output: tiled zarr arrays saved to disk
3. **Build pytorch dataloader**
   - Input: path to tiled zarr arrays, batch size
   - Output: pytorch dataloader object
4. **Train masked autoencoder model**
   - Input: pytorch dataloader object, number of epochs, learning rate
   - Output: trained masked autoencoder model saved to disk
5. **Encode all tiles using trained encoder**
   - Input: path to tiled zarr arrays, trained masked autoencoder model
   - Output: encoded tiles saved to disk
6. **Run kmeans clustering on tiles**
   - Input: encoded tiles, number of clusters
   - Output: cluster number for each tile saved to disk
7. **Map cluster color back onto original slide**
   - Input: path to original slide, cluster number for each tile
   - Output: original slide with cluster color mapping saved to disk

Let me know if you need any further assistance!