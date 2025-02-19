#!/usr/bin/env python
"""
Usage:
  python preprocess.py <root_path> <mat_path> <output_path> <tile_size> <mode> <pipeline>

Parameters:
  root_path  : Directory containing .tif files.
  mat_path   : Directory containing .mat files (only used for SCME).
  output_path: Directory where outputs will be saved.
  tile_size  : Tile size (integer).
  mode       : 'full' or 'subsample' (used for SCME).
  pipeline   : 'SCME' or 'LTME'.
"""

import sys
import os
from scipy.io import loadmat
import tifffile as tf

# Import the relevant functions from our modules
from io import scme_qptiff_to_zarr, ltme_qptiff_to_zarr
from tile import scme_gen_tiles, ltme_gen_tiles

def main():
    if len(sys.argv) < 7:
        print("Usage: python preprocess.py <root_path> <mat_path> <output_path> <tile_size> <mode> <pipeline>")
        sys.exit(1)
    
    root_path = sys.argv[1]
    mat_path = sys.argv[2]    # Only used for SCME
    output_path = sys.argv[3]
    tile_size = int(sys.argv[4])
    mode = sys.argv[5]        # 'full' or 'subsample' (SCME only)
    pipeline = sys.argv[6].upper()  # Either 'SCME' or 'LTME'
    
    for file in os.listdir(root_path):
        if file.endswith('.tif'):
            input_file = os.path.join(root_path, file)
            print(f"Processing {file} using {pipeline} pipeline...")
            
            if pipeline == "SCME":
                # Build output paths and load corresponding MAT file.
                output_file_base = file.replace('.tif', '')
                output_file_path = os.path.join(output_path, output_file_base)
                mat_file = os.path.join(mat_path, output_file_base + ".mat")
                mat_data = loadmat(mat_file)
                # Create Zarr array using the SCME function.
                zarr_array = scme_qptiff_to_zarr(input_file, output_path, mat_data)
                # Read image for composite creation.
                image = tf.imread(input_file)
                scme_gen_tiles(image, zarr_array, mat_data, tile_size, output_file_path, file.replace('.tif', '.png'), mode)
            
            elif pipeline == "LTME":
                zarr_array = ltme_qptiff_to_zarr(input_file, output_path)
                ltme_gen_tiles(zarr_array, tile_size, None)
            
            else:
                print(f"Invalid pipeline specified: {pipeline}. Must be SCME or LTME.")
                sys.exit(1)
            
            print(f"Finished processing {file}")

if __name__ == '__main__':
    main()
