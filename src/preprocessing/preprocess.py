# Preprocess all data from a directory
import sys
import os
sys.path.append('/gpfs/scratch/ss14424/CANVAS-ss/src')
from preprocessing import io, tile

def main():
    root_path = sys.argv[1]
    output_path = sys.argv[2]
    if len(sys.argv) > 3:
        tile_size = int(sys.argv[3])
    else:
        tile_size = 64
    # List all files end with .qptiff
    for file in os.listdir(root_path):
        if file.endswith(".tif"):
            input_file = os.path.join(root_path, file)
            print(f"Processing {file}")
            zarr_array = io.qptiff_to_zarr(input_file, output_path)
            output_file_path = os.path.join(output_path, 
                                            file.replace('.tif', ''))
            tile.gen_tiles(zarr_array, tile_size, output_file_path)
            print(f"Finished processing {file}")

if __name__ == '__main__':
    main()