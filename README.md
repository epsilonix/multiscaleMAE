# Multi-scale MAE pipeline
For analysis of multiplexed images

1. **Use preprocessing/run_preprocessing.sh to convert tif files to zarr arrays and generate tiles**
   - io.py takes tif files and converts them to zarr arrays.  
   - tile.py takes zarr arrays and generates tiles of the desired size. 
   - LTME tiles are 64x64 pixels, and exclude regions with low signal.
   - SCME tiles are 16x16 pixels and are centered on individual cells, without any masking.
   - The optional subsample parameter in run_preprocessing.sh is used to subsample the tiles for SCME, to reduce the number of tiles needed for model training.
   - OPTIONAL: consolidator.py consolidates the channels of the images into a single image with the target channel structure.
2. **Use model/pretrain.sh to train the masked autoencoder model**
   - Input parameters are the path to the tiled zarr arrays, the tile size used to generate the tiles, desired batch size, and number of training epochs..
   - Be sure to select pipeline (SCME for single cell microenvironments, LTME for local tumor microenvironments) 
3. **Use inference/canvas_base_run.sh to perform inference on all tiles using trained encoder**
   - Input parameters are the pipeline type (SCME or LTME), the path to the trained model, and the path to the tiled zarr arrays.
   - Outputs numpy files for tile embeddings, tile location, sample name, image mean, and cluster.
4. **Use the Jupyter notebooks in analysis/ to perform downstream analysis**
   - ltme_analysis.ipynb generates the LTME composition map and UMAP
   - scme_analysis.ipynb generates the SCME composition map and UMAP
   - ltme_l2_bcell_survival.ipynb performs the LTME L2 and B cell survival analysis

For recommended hardware settings and estimated runtimes, please refer to the sbatch files.

