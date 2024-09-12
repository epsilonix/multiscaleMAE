import numpy as np
import pandas as pd
import os
import torch.utils.data as data
from torchvision import transforms
from skimage.io import imsave, imread
from skimage.transform import resize
from skimage.draw import polygon
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
        
        # To control printing for only one image
        self.debug_printed = False

        
    def __getitem__(self, index):
        # Load the original image tile based on the position
        x, y = self.tile_pos[index][0], self.tile_pos[index][1]
        image = self.read_region(x, y, self.tile_size, self.tile_size)
        
        # Apply boundary mask if blankoutbg is set to True
        if self.blankoutbg:
            boundary = self.boundary[index]  # Assuming boundary is in the format of a mask or coordinates
            tile_pos = (x, y)

            # Print the number of zeros before and after masking for the first image
            if not self.debug_printed and index == 0:  # Modify index to control which image to debug
                num_zeros_before = np.sum(image == 0)
                print(f"Number of zeros before transformation at index {index}: {num_zeros_before}")
                
                # Apply the mask
                image = self.apply_boundary_mask(image, boundary, tile_pos)
                
                num_zeros_after = np.sum(image == 0)
                print(f"Number of zeros after transformation at index {index}: {num_zeros_after}")
                
                self.debug_printed = True  # Ensure this prints only once
            
        # Apply transformations if any
        if self.transform is not None:
            transformed_image = self.transform(image)
        else:
            transformed_image = transforms.ToTensor()(image)

        label = None
        x = self.tile_pos[index][0]
        y = self.tile_pos[index][1]
        img_id = index
        
        return transformed_image, label, x, y, img_id
    
    def apply_boundary_mask(self, image, boundary, tile_pos):
        tile_x, tile_y = tile_pos  # Tile's upper-left corner coordinates

        # Convert the boundary string to a list of [x, y] coordinate pairs
        boundary = json.loads(boundary)  # Convert string representation to a list of lists

        # Adjust coordinates relative to the tile's position
        adjusted_boundary = [(x - tile_x, y - tile_y) for x, y in boundary]

        # Separate x and y coordinates for creating the polygon mask
        rr, cc = zip(*adjusted_boundary)

        # Create a blank mask with the same size as the image
        mask = np.zeros((self.tile_size, self.tile_size), dtype=np.uint8)

        # Create a polygon mask using the boundary coordinates
        rr, cc = polygon(cc, rr, mask.shape)  # Notice rr and cc are swapped to match (y, x) format
        mask[rr, cc] = 1  # Set mask pixels corresponding to the boundary

        # Apply the mask to the image
        masked_image = image * mask[..., np.newaxis]

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
