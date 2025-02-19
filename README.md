# Multi-scale MAE pipeline
For analysis of multiplexed images

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