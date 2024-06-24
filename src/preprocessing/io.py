# Converting QPTIFF to Zarr with channel metadata
import os
import xml.etree.ElementTree as ET
import tifffile
import zarr
import numpy as np

def qptiff_to_zarr(input_file, output_root, boundaries_file, chunk_size=(None, 256, 256)):
    channels_to_exclude = [0, 5, 11, 13]  # Define the channels to exclude

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

    # Read the boundaries file and create a mask
    boundaries = read_boundaries(boundaries_file)
    mask = create_mask(boundaries, img_data.shape[1:])

    # Apply the mask to the image data
    img_data = apply_mask(img_data, mask)

    # Create the Zarr array
    z_arr = zarr.array(img_data, chunks=chunk_size, store=output_zarr)
    return z_arr

def read_boundaries(boundaries_file):
    # Implement the function to read the boundaries from the boundaries_file
    # This function should return a list of boundary coordinates
    boundaries = []
    tree = ET.parse(boundaries_file)
    root = tree.getroot()
    
    for boundary in root.findall('.//Boundary'):
        coords = []
        for point in boundary.findall('.//Point'):
            x = int(point.get('X'))
            y = int(point.get('Y'))
            coords.append((x, y))
        boundaries.append(coords)
    
    return boundaries

def create_mask(boundaries, shape):
    mask = np.zeros(shape, dtype=bool)
    for boundary in boundaries:
        boundary_array = np.array(boundary)
        rr, cc = polygon(boundary_array[:, 1], boundary_array[:, 0], mask.shape)
        mask[rr, cc] = True
    return mask

def apply_mask(img_data, mask):
    for c in range(img_data.shape[0]):
        img_data[c] *= mask
    return img_data
