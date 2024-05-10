import zarr
import os

def check_tensor_dimensions(base_path, expected_shape=(17, 224, 224)):
    """
    Walk through directories to find zarr files and check their tensor dimensions.
    """
    # Walk through all subdirectories in the base path
    for root, dirs, files in os.walk(base_path):
        for file in files:
            # Check if the file is a Zarr file (assuming naming or extension here, adjust as necessary)
            if file.endswith('.zarr'):
                file_path = os.path.join(root, file)
                try:
                    # Load the Zarr file
                    z = zarr.open(file_path, mode='r')
                    # Get the shape of the dataset, assuming it is stored directly at the root of the Zarr file
                    shape = z.shape

                    # Check if the shape matches the expected dimensions
                    if shape != expected_shape:
                        print(f"Dimension mismatch in {file_path}: expected {expected_shape}, found {shape}")
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

if __name__ == '__main__':
    base_path = '/gpfs/scratch/ss14424/Brain/cells/img_output_10'
    check_tensor_dimensions(base_path)
