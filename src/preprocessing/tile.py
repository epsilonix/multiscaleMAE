# 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
# Generating tiles for a given zarr image
import os
import zarr
import numpy as np
from skimage.measure import block_reduce
from skimage.transform import resize
from skimage.io import imsave

def gen_tiles(slide: str, tile_size: int = 128, 
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
    print('Generating mask...')
    mask = gen_mask(thumbnail)
    save_img(output_path, 'mask', tile_size // 4, mask)
    # Generate and save tile positions
    print('Generating tile positions...')
    tile_img, positions = gen_tile_positions(slide, mask, tile_size=tile_size)
    save_img(output_path, 'tile_img', tile_size, tile_img)
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

def gen_mask(thumbnail: np.ndarray, threshold: int = 0.5) -> np.ndarray:
    ''' Generate mask for a given thumbnail '''
    mask = np.where(thumbnail > threshold, 1, 0)
    return mask

def gen_tile_positions(slide: zarr, mask: np.ndarray, tile_size: int = 128, 
              threshold: float = 0.1) -> np.ndarray:
    ''' Generate tiles for a given slide and mask '''
    # Read numpy dimensions
    _, slide_height, slide_width = slide.shape
    grid_height, grid_width = slide_height // tile_size, slide_width // tile_size
    # Check propertion are correct
    print(f"Slide Height: {slide_height} Width: {slide_width} Aspect Ratio: {slide_height / slide_width}")
    print(f"Mask Aspect Ratio: {mask.shape[0] / mask.shape[1]}")
    print(abs(mask.shape[0] / mask.shape[1] - slide_height / slide_width))
    
    aspect_ratio_diff = abs(mask.shape[0] / mask.shape[1] - slide_height / slide_width)
    if aspect_ratio_diff >= 0.05:
        print(f"Aspect ratio difference too large: {aspect_ratio_diff}... skipping")
    else:
        # Convert mask to pixel level grid
        mask = resize(mask, (grid_height, grid_width), order=0, anti_aliasing=False)
        # Generate mask
        tile_img = np.where(mask > threshold, 1, 0)
        # Generate tiles
        hs, ws = np.where(mask > threshold)
        positions = np.array(list(zip(hs, ws))) * tile_size
        print(f'Tile Image:{tile_img}, Positions:{positions}')
        return tile_img, positions
