import os
import tifffile
import zarr
import numpy as np
from skimage.draw import polygon

def qptiff_to_zarr(input_file, output_root, mat_data, chunk_size=(None, 256, 256)):
    channels_to_exclude = []  # Define the channels to exclude

    # Check if output file already exists
    output_path = os.path.join(output_root, os.path.basename(input_file).replace('.tif', ''))
    output_zarr = output_path + '/data.zarr'
    print(f'Create the directory')

    os.makedirs(output_path, exist_ok=True)

    # Check if Zarr dataset already exists at the path
    if os.path.exists(output_zarr):
        print(f"Zarr dataset already exists at {output_zarr}. Consider handling this scenario appropriately.")
        return zarr.open(output_zarr, mode='a')  # Open in append mode as a placeholder action

    # Read the QPTIFF file and exclude specific channels
    img_data_slices = []
    print(f'Read the tif file')
    with tifffile.TiffFile(input_file) as tif:
        for i, page in enumerate(tif.pages):
            if i not in channels_to_exclude:
                img_data_slices.append(page.asarray())
                print(f'Included channel {i} with shape {page.shape}')
            else:
                print(f'Excluded channel {i}')

    # Stack the included image data slices along the first axis
    if img_data_slices:
        img_data = np.stack(img_data_slices, axis=0)
        print(f'input file {input_file} has shape {img_data.shape}')
    else:
        print("No valid image data found after excluding specified channels.")
        return None


    # Create the Zarr array
    z_arr = zarr.array(img_data, chunks=chunk_size, store=output_zarr)
    return z_arr

