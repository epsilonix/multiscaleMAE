import os
import zarr

# Define the base path where your folders with `.zarr` files are located
base_path = '/gpfs/scratch/ss14424/Brain/cells/img_output_10/'

# Define the expected shape
expected_shape = (17, 224, 224)

# Iterate over each folder in the base path
for folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder)
    if os.path.isdir(folder_path):
        # Find all `.zarr` files in this folder
        zarr_files = [file for file in os.listdir(folder_path) if file.endswith('.zarr')]
        # Check each `.zarr` file in this folder
        for zarr_file in zarr_files:
            zarr_file_path = os.path.join(folder_path, zarr_file)
            try:
                # Load the Zarr array
                zarr_array = zarr.open(zarr_file_path, mode='r')
                # Check the shape
                if zarr_array.shape != expected_shape:
                    print(f"Mismatch in file {zarr_file_path}: expected {expected_shape}, found {zarr_array.shape}")
                else:
                    print(f"File {zarr_file_path} has the correct shape.")
                print(f"Successfully loaded Zarr file: {zarr_file_path}")
            except Exception as e:
                print(f"Error checking file {zarr_file_path}: {str(e)}")
        # Print message indicating the completion of processing for this folder
        print(f"Finished checking all Zarr files in folder {folder}.")
