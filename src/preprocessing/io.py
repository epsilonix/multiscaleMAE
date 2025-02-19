import os
import tifffile as tf
import zarr
import numpy as np

def scme_qptiff_to_zarr(input_file, output_root, mat_data, chunk_size=(None, 256, 256)):
    """
    SCME pipeline: Convert QPTIFF to Zarr with additional MAT data.
    """
    channels_to_exclude = []  # Define channels to exclude as needed
    output_path = os.path.join(output_root, os.path.basename(input_file).replace('.tif', ''))
    output_zarr = output_path + '/data.zarr'
    print(f"[SCME] Creating directory: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    
    if os.path.exists(output_zarr):
        print(f"[SCME] Zarr dataset already exists at {output_zarr}. Opening in append mode.")
        return zarr.open(output_zarr, mode='a')
    
    img_data_slices = []
    print("[SCME] Reading tif file...")
    with tf.TiffFile(input_file) as tif:
        for i, page in enumerate(tif.pages):
            if i not in channels_to_exclude:
                img_data_slices.append(page.asarray())
                print(f"[SCME] Included channel {i} with shape {page.shape}")
            else:
                print(f"[SCME] Excluded channel {i}")
    
    if img_data_slices:
        img_data = np.stack(img_data_slices, axis=0)
        print(f"[SCME] Input file {input_file} has shape {img_data.shape}")
    else:
        print("[SCME] No valid image data found after excluding specified channels.")
        return None
    
    z_arr = zarr.array(img_data, chunks=chunk_size, store=output_zarr)
    return z_arr

def ltme_qptiff_to_zarr(input_file, output_root, chunk_size=(None, 256, 256)):
    """
    LTME pipeline: Convert QPTIFF to Zarr without needing MAT data.
    """
    channels_to_exclude = []
    output_path = os.path.join(output_root, os.path.basename(input_file).replace('.tif', ''))
    output_zarr = output_path + '/data.zarr'
    os.makedirs(output_path, exist_ok=True)
    
    img_data_slices = []
    with tf.TiffFile(input_file) as tif:
        for i, page in enumerate(tif.pages):
            if i not in channels_to_exclude:
                img_data_slices.append(page.asarray())
    img_data = np.stack(img_data_slices, axis=0)
    print(f"[LTME] Input file {input_file} has shape {img_data.shape}")
    z_arr = zarr.array(img_data, chunks=chunk_size, store=output_zarr)
    return z_arr
