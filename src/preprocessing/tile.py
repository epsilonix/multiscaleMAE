# 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
# Generating tiles for a given zarr image
import os
import zarr
import numpy as np
from skimage.measure import block_reduce
from skimage.transform import resize
from skimage.io import imsave
from scipy.io import loadmat
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

def calculate_polygon_area(coords):
    """Calculate the area of a polygon using the shoelace formula.

    Args:
    - coords (list of tuples): List of (x, y) tuples for each vertex of the polygon.

    Returns:
    - float: The area of the polygon.
    """
    x = np.array([coord[0] for coord in coords])
    y = np.array([coord[1] for coord in coords])

    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def gen_tiles(slide: str, mat_data, tile_size: int = 128, 
              output_path: str = None) -> np.ndarray:
    ''' Generate tiles for a given slide '''
    if output_path is None: output_path = f'{os.path.dirname(slide)}'
    output_path += '/tiles/'
    os.makedirs(output_path, exist_ok=True)
    # Read slide
    print('Reading slide...')
    print(f'IMG has a dimension of {slide.shape}')
    if isinstance(slide, str):
        slide = zarr.load(slide)
    print(f'ZARR IMG has a dimension of {slide.shape}')
    # Generate and save thumbnail
    print('Generating thumbnail...')
    thumbnail = gen_thumbnail(slide, scaling_factor=tile_size // 4)
    save_img(output_path, 'thumbnail', tile_size // 4, thumbnail)
    # Generate and save mask

    boundaries_info = mat_data['Boundaries']

    # Correct dimensions of the matrix
    correct_dimensions = image.shape[1:3][::-1]

    # Initialize a list to hold the coordinates for all boundaries
    all_boundaries_coords = []

    for i in range(boundaries_info.shape[1]):
        # Extract linear indices for the current boundary
        linear_indices = boundaries_info[0, i].flatten()  # Ensure it's a 1D array

        # Convert linear indices to x and y coordinates
        x_coords, y_coords = np.unravel_index(linear_indices, correct_dimensions)

        # Store the coordinates as tuples in the list
        boundary_coords = list(zip(x_coords, y_coords))

        # Calculate the area of the boundary
        area = calculate_polygon_area(boundary_coords)

        # Only add the boundary if its area is <= 100
        if area <= 144:
            all_boundaries_coords.append(boundary_coords)

    # Select 20% of the coordinates randomly
    sample_boundaries_coords = random.sample(all_boundaries_coords, 200)
    
    centroid_coords = []

    for boundary_coords in sample_boundaries_coords:
        # Convert list of tuples to a numpy array for easy mathematical operations
        boundary_array = np.array(boundary_coords)

        # Calculate the mean of x and y coordinates
        centroid_x = np.mean(boundary_array[:, 0])
        centroid_y = np.mean(boundary_array[:, 1])

        # Append the centroid coordinates as a tuple to the list
        centroid_coords.append((centroid_x, centroid_y))

    centroid_coords = sorted(centroid_coords, key=lambda x: (x[0], x[1]))
    centroid_coords = [(round(x, 3), round(y, 3)) for x, y in centroid_coords]
    
    # Size of the square centered on each centroid
    half_side_length = 5  # Half the side length of the square, for a total side length of 10

    positions = []

    # Plot a square centered on each centroid in `centroid_coords`
    for centroid_x, centroid_y in centroid_coords:
        # Calculate the bottom-left corner of the square
        bottom_left_x = centroid_x - half_side_length
        bottom_left_y = centroid_y - half_side_length

        positions.append((bottom_left_y, bottom_left_x))
 
    with open(os.path.join(output_path, f'positions_{tile_size}.csv'), 'w') as f:
        f.write(' ,h,w\n')
        for i, (h, w) in enumerate(positions):
            f.write(f'{i},{h},{w}\n')
    print(f'Generated {len(positions)} tiles for slide with shape {slide.shape}')



def save_img(output_path: str, task: str, tile_size: int, img: np.ndarray):
    ''' Save image to output path '''
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    imsave(os.path.join(output_path, f'{task}_{tile_size}.png'), img)

def gen_thumbnail(slide: zarr, scaling_factor: int) -> np.ndarray:
    ''' Generate thumbnail for a given slide '''
    # Make sure first channel is smaller than the second and third
    assert slide.shape[0] < slide.shape[1] and slide.shape[0] < slide.shape[2]
    cache = block_reduce(slide, 
                         block_size=(slide.shape[0], 
                                     scaling_factor, scaling_factor), 
                         func=np.mean)
    # Remove bright pixels top 5 percentile
    cache = np.clip(cache, 0, np.percentile(cache, 95))
    cache /= cache.max()
    thumbnail = np.clip(cache, 0, 1).squeeze()
    return thumbnail

