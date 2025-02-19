import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import block_reduce
from skimage.io import imsave
from skimage.transform import resize
import zarr

def scme_gen_tiles(image, zarr_array, mat_data, tile_size, output_file_path, image_filename, mode='full'):
    """
    SCME pipeline: Generate composite image and tile positions based on boundaries from MAT file.
    """
    # Define colors and common cell types
    colors = [
        (0, 0, 139), (173, 216, 230), (135, 206, 250), (212, 226, 228),
        (70, 130, 180), (149, 229, 68), (25, 25, 112), (228, 82, 50),
        (255, 165, 0), (176, 226, 219), (255, 140, 0), (255, 127, 80),
        (228, 71, 184), (255, 99, 71), (255, 182, 193), (176, 226, 219),
        (73, 52, 229), (0, 255, 255), (255, 255, 0), (238, 130, 238),
        (50, 205, 50), (255, 20, 147), (186, 85, 211), (0, 128, 0),
        (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
        (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
        (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
        (0, 0, 0), (0, 0, 0)
    ]
    very_common_cell_types = {'Cancer', 'Astrocytes', 'Cl BMDM', 'Alt BMDM', 'Endothelial cell', 'Cl MG', 'Alt MG'}

    print(f"[SCME] Zarr array type: {type(zarr_array)}")
    if output_file_path is None:
        output_file_path = os.path.dirname(zarr_array)
    output_tiles_path = os.path.join(output_file_path, 'tiles')
    os.makedirs(output_tiles_path, exist_ok=True)
    
    # Extract boundaries and cell types from MAT data
    boundaries_info = mat_data['Boundaries']
    cell_types = mat_data['cellTypes']
    # Determine correct dimensions (width, height) from image shape
    correct_dimensions = image.shape[1:3][::-1]
    
    all_boundaries_coords = []
    for i in range(boundaries_info.shape[1]):
        linear_indices = boundaries_info[0, i].flatten()
        x_coords, y_coords = np.unravel_index(linear_indices, correct_dimensions)
        boundary_coords = list(zip(x_coords, y_coords))
        # Calculate area using the shoelace formula
        def calculate_polygon_area(coords):
            x = np.array([pt[0] for pt in coords])
            y = np.array([pt[1] for pt in coords])
            return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        area = calculate_polygon_area(boundary_coords)
        cell_type = cell_types[i][0][0] if cell_types[i][0].size > 0 else 'Unknown'
        if 20 < area < 200:
            all_boundaries_coords.append((boundary_coords, cell_type))
    
    num_channels, height, width = image.shape
    brightness_factor = 3.0
    composite_image = np.zeros((height, width, 3), dtype=np.float32)
    global_max = np.max(image)
    for i in range(num_channels):
        color = colors[i] if i < len(colors) else (255, 255, 255)
        channel_normalized = image[i, :, :] / global_max
        for j in range(3):
            composite_image[:, :, j] += channel_normalized * color[j]
    composite_image = np.clip(composite_image / np.max(composite_image, axis=(0, 1)) * brightness_factor, 0, 1)
    
    plt.figure(figsize=(15,15))
    plt.imshow(composite_image)
    
    positions = []
    sampled_cell_counts = {cell_type: 0 for cell_type in very_common_cell_types}
    sample_boundaries_coords = []
    if mode == 'subsample':
        print("[SCME] Subsampling mode activated...")
        cell_type_groups = {cell_type: [] for cell_type in very_common_cell_types}
        other_cell_types = []
        for boundary_info in all_boundaries_coords:
            boundary_coords, cell_type = boundary_info
            if cell_type in very_common_cell_types:
                cell_type_groups[cell_type].append(boundary_info)
            else:
                other_cell_types.append(boundary_info)
        for cell_type, boundaries in cell_type_groups.items():
            sample_count = min(len(boundaries), 100)
            if boundaries:
                sampled_boundaries = random.sample(boundaries, sample_count)
                sample_boundaries_coords.extend(sampled_boundaries)
                sampled_cell_counts[cell_type] += sample_count
        sample_boundaries_coords.extend(other_cell_types)
        print(f"[SCME] Generated {len(sample_boundaries_coords)} boundaries after subsampling.")
    else:
        print("[SCME] Full mode activated...")
        sample_boundaries_coords = all_boundaries_coords.copy()
        print(f"[SCME] Generated {len(sample_boundaries_coords)} boundaries.")
    
    for boundary_info in sample_boundaries_coords:
        boundary_coords, cell_type = boundary_info
        boundary_array = np.array(boundary_coords)
        plt.plot(boundary_array[:, 0], boundary_array[:, 1], color='cyan', linewidth=0.5)
        centroid_x = int(np.round(np.mean(boundary_array[:, 0])))
        centroid_y = int(np.round(np.mean(boundary_array[:, 1])))
        half_side_length = tile_size / 2
        if 10 <= centroid_x <= width - 10 and 10 <= centroid_y <= height - 10:
            positions.append((centroid_y - half_side_length, centroid_x - half_side_length, cell_type, json.dumps(boundary_coords)))
    
    positions_file = os.path.join(output_tiles_path, f'positions_{tile_size}.csv')
    with open(positions_file, 'w') as f:
        f.write(" ,h,w,celltype,boundary\n")
        for i, (h, w, celltype, boundary) in enumerate(positions):
            f.write(f"{i},{h},{w},{celltype},\"{boundary}\"\n")
    print(f"[SCME] Generated {len(positions)} tiles for image with shape {image.shape}")

def ltme_gen_tiles(slide, tile_size=128, output_path=None):
    """
    LTME pipeline: Generate thumbnail, mask, and tile positions.
    """
    if output_path is None:
        output_path = os.path.dirname(slide)
    output_tiles_path = os.path.join(output_path, 'tiles')
    os.makedirs(output_tiles_path, exist_ok=True)
    
    print("[LTME] Reading slide...")
    if isinstance(slide, str):
        slide = zarr.load(slide)
    print(f"[LTME] Slide shape: {slide.shape}")
    
    # Generate thumbnail
    print("[LTME] Generating thumbnail...")
    thumbnail = ltme_gen_thumbnail(slide, scaling_factor=tile_size // 4)
    ltme_save_img(output_tiles_path, 'thumbnail', tile_size // 4, thumbnail)
    
    # Generate mask
    print("[LTME] Generating mask...")
    mask = ltme_gen_mask(thumbnail)
    ltme_save_img(output_tiles_path, 'mask', tile_size // 4, mask)
    
    # Generate tile positions
    print("[LTME] Generating tile positions...")
    tile_img, positions = ltme_gen_tile_positions(slide, mask, tile_size=tile_size)
    ltme_save_img(output_tiles_path, 'tile_img', tile_size, tile_img)
    
    positions_file = os.path.join(output_tiles_path, f'positions_{tile_size}.csv')
    with open(positions_file, 'w') as f:
        f.write(" ,h,w\n")
        for i, (h, w) in enumerate(positions):
            f.write(f"{i},{h},{w}\n")
    print(f"[LTME] Generated {len(positions)} tiles for slide with shape {slide.shape}")

# --- LTME Helper Functions ---

def ltme_save_img(output_path, task, tile_size, img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    imsave(os.path.join(output_path, f"{task}_{tile_size}.png"), img)

def ltme_gen_thumbnail(slide, scaling_factor):
    # Ensure slide dimensions are as expected
    assert slide.shape[0] < slide.shape[1] and slide.shape[0] < slide.shape[2]
    print(f"[LTME] Generating thumbnail with scaling factor {scaling_factor}...")
    numpy_slide = slide[...]
    cache = block_reduce(numpy_slide, block_size=(slide.shape[0], scaling_factor, scaling_factor), func=np.mean)
    cache = np.clip(cache, 0, np.percentile(cache, 95))
    cache /= cache.max()
    thumbnail = np.clip(cache, 0, 1).squeeze()
    return thumbnail

def ltme_gen_mask(thumbnail, threshold=0.5):
    mask = np.where(thumbnail > threshold, 1, 0)
    return mask

def ltme_gen_tile_positions(slide, mask, tile_size=128, threshold=0.1):
    _, slide_height, slide_width = slide.shape
    grid_height, grid_width = slide_height // tile_size, slide_width // tile_size
    print(f"[LTME] Slide Height: {slide_height}, Width: {slide_width}, Aspect Ratio: {slide_height/slide_width}")
    print(f"[LTME] Mask Aspect Ratio: {mask.shape[0] / mask.shape[1]}")
    aspect_ratio_diff = abs(mask.shape[0] / mask.shape[1] - slide_height / slide_width)
    if aspect_ratio_diff >= 0.05:
        print(f"[LTME] Aspect ratio difference too large: {aspect_ratio_diff}... skipping")
        return None, []
    else:
        mask_resized = resize(mask, (grid_height, grid_width), order=0, anti_aliasing=False)
        tile_img = np.where(mask_resized > threshold, 1, 0)
        hs, ws = np.where(mask_resized > threshold)
        positions = np.array(list(zip(hs, ws))) * tile_size
        return tile_img, positions
