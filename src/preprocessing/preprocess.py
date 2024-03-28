# Preprocess all data from a directory
import sys
import os
sys.path.append('/gpfs/scratch/ss14424/canvas-sc/src')
from preprocessing import io, tile
from scipy.io import loadmat

def main():
    root_path = sys.argv[1]
    mat_path = sys.argv[2]
    output_path = sys.argv[3]
    if len(sys.argv) > 4:
        tile_size = int(sys.argv[4])
    else:
        tile_size = 10
    # List all files end with .qptiff
    for file in os.listdir(root_path):
        if file.endswith(".tif"):
            input_file = os.path.join(root_path, file)
            print(f"Processing {file}")
            zarr_array = io.qptiff_to_zarr(input_file, output_path)
            output_file_base = file.replace('.tif', '')
            output_file_path = os.path.join(output_path, output_file_base)
            print(f'Obtaining mat for {file}')
            mat_dir = os.path.join(mat_path, output_file_base + ".mat")
            mat_data = loadmat(mat_dir)
            
            tile.gen_tiles(input_file, zarr_array, mat_data, tile_size, output_file_path)
            print(f"Finished processing {file}")

if __name__ == '__main__':
    main()