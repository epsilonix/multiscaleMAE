# Preprocess all data from a directory
import sys
import os
sys.path.append('/gpfs/scratch/ss14424/singlecell/src')
from preprocessing import io, tile
from scipy.io import loadmat
import tifffile as tf

def main():
    root_path = sys.argv[1]
    mat_path = sys.argv[2]
    output_path = sys.argv[3]
    tile_size = int(sys.argv[4])
    mode = sys.argv[5]  # Add the mode argument

    # List all files ending with .tif
    for file in os.listdir(root_path):
        if file.endswith(".tif"):
            
            # Opening file
            input_file = os.path.join(root_path, file)
            image = tf.imread(input_file)
            
            # Obtain mat file
            print(f'Obtaining mat for {file}')
            output_file_base = file.replace('.tif', '')
            output_file_path = os.path.join(output_path, output_file_base)
            mat_dir = os.path.join(mat_path, output_file_base + ".mat")
            mat_data = loadmat(mat_dir)
            
            # io
            zarr_array = io.qptiff_to_zarr(input_file, output_path, mat_data)
            
            # Tile the image
            image_filename = file.replace('.tif', '.png')
            tile.gen_tiles(image, zarr_array, mat_data, tile_size, output_file_path, image_filename, mode)  # Pass the mode argument
            print(f"Finished processing {file}")

if __name__ == '__main__':
    main()
