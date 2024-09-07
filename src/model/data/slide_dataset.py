import numpy as np
import pandas as pd
import os
import torch.utils.data as data
from torchvision import transforms
from skimage.io import imsave, imread
from skimage.transform import resize
import json

class SlideDataset(data.Dataset):
    ''' Dataset for slides '''

    def __init__(self, root_path=None, tile_size=None, transform=None, blankoutbg=False):
        ''' 
        Initialize the dataset 
        root_path: root path of the dataset (for saving processed file purposes)
        '''
        self.root_path = root_path
        self.tile_size = tile_size
        self.transform = transform
        self.blankoutbg = blankoutbg  

        

        self.df = self.load_tile_data(tile_size)
        # Load tiles positions from disk
        self.tile_pos = self.load_tiles()
        self.celltypes = self.load_celltypes()
        self.boundary = self.load_boundary()

    def __getitem__(self, index):
        # Load the original image tile based on the position
        image = self.read_region(self.tile_pos[index][0], self.tile_pos[index][1], self.tile_size, self.tile_size)
        
        # Apply boundary mask if blankoutbg is set to True
        if self.blankoutbg:
            boundary = self.boundary[index]  # Assuming boundary is in the format of a mask or coordinates
            image = self.apply_boundary_mask(image, boundary)

        # Apply transformations if any
        if self.transform is not None:
            transformed_image = self.transform(image)
        else:
            transformed_image = transforms.ToTensor()(image)

        # Additional labels or information
        label = None
        x = self.tile_pos[index][0]
        y = self.tile_pos[index][1]
        img_id = index
        
        return transformed_image, label, x, y, img_id
    
    def apply_boundary_mask(self, image, boundary):
        # Convert the boundary information into a mask or crop coordinates
        # For example, if boundary is a binary mask of the same size as the tile:
        if isinstance(boundary, np.ndarray) and boundary.shape == image.shape[:2]:
            # Apply the mask directly to the image
            masked_image = image * boundary[..., np.newaxis]
        else:
            # If boundary is given as coordinates, adjust extraction
            min_x, min_y, max_x, max_y = boundary  # Assuming boundary is (min_x, min_y, max_x, max_y)
            masked_image = image[min_y:max_y, min_x:max_x]

        return masked_image

    def __len__(self):
        return len(self.tile_pos)

    def read_slide(self, root_path):
        ''' Read slide from disk'''
        raise NotImplementedError

    def read_region(self, pos_x, pos_y, width, height):
        ''' x and y are the coordinates of the top left corner '''
        raise NotImplementedError

    def get_slide_dimensions(self):
        ''' Get slide dimensions '''
        raise NotImplementedError

    def save_thumbnail(self):
        ''' Save a thumbnail of the slide '''
        raise NotImplementedError

    def load_tile_data(self, tile_size):
        ''' Load the tile data from disk and save it as a DataFrame '''
        tile_path = f'{self.root_path}/tiles/positions_{tile_size}.csv'
        df = pd.read_csv(tile_path)
        return df

    def load_tiles(self):
        ''' Extract tile positions from the loaded DataFrame '''
        tile_pos = self.df[["h", "w"]].to_numpy()
        return tile_pos

    def load_celltypes(self):
        ''' Extract cell types from the loaded DataFrame '''
        celltypes = self.df["celltype"].values
        return celltypes

    def load_boundary(self):
        ''' Extract boundary information from the loaded DataFrame '''
        boundary = self.df["boundary"].values
        return boundary

    # Generate tiles from mask
    def load_tiling_mask(self, mask_path, tile_size):
        ''' Load tissue mask to generate tiles '''
        # Get slide dimensions
        slide_width, slide_height = self.get_slide_dimensions()
        # Specify grid size
        grid_width, grid_height = slide_width // tile_size, slide_height // tile_size
        # Create mask
        if mask_path is not None: # Load mask from existing file
            mask_temp = np.array(imread(mask_path)).swapaxes(0, 1)
            assert abs(mask_temp.shape[0] / mask_temp.shape[1] - slide_width / slide_height) < 0.01, 'Mask shape does not match slide shape'
            # Convert mask to patch-pixel level grid
            mask = resize(mask_temp, (grid_width, grid_height), anti_aliasing=False)
        else:
            mask = np.ones((grid_width, grid_height)) # Tile all regions
        return mask

    def generate_tiles(self, tile_size, mask_path=None, mask_id='default', threshold=0.99):
        ''' 
        Generate tiles from a slide
        threshold: minimum percentage of tissue mask in a tile
        '''
        # Load mask
        mask = self.load_tiling_mask(mask_path, tile_size)
        # Generate tile coordinates according to masked grid
        ws, hs = np.where(mask >= threshold)
        positions = (np.array(list(zip(ws, hs))) * tile_size)
        # Save tile top left positions
        tile_path = f'{self.root_path}/tiles/{mask_id}'
        save_path = f'{tile_path}/{tile_size}'
        os.makedirs(save_path, exist_ok=True)
        np.save(f'{save_path}/tile_positions_top_left.npy', positions)
        # Save mask image
        mask_img = np.zeros_like(mask)
        mask_img[ws, hs] = 1
        imsave(f'{save_path}/mask.png', (mask_img.swapaxes(0, 1) * 255).astype(np.uint8))
