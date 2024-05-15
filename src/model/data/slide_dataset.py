import numpy as np
import pandas as pd
import os
import torch.utils.data as data
from torchvision import transforms

from skimage.io import imsave, imread
from skimage.transform import resize

class SlideDataset(data.Dataset):
    ''' Dataset for slides '''

    def __init__(self, root_path=None, tile_size=None, transform=None):
        ''' 
        Initialize the dataset 
        root_path: root path of the dataset (for saving processed file purposes)
        '''
        self.root_path = root_path
        self.tile_size = tile_size
        
        print(f'slide_dataset thinks tile_size is {tile_size}')
        
        self.transform = transform

        if tile_size is not None:
            # Load tiles positions from disk
            self.tile_pos = self.load_tiles(tile_size)
            self.valid_cell_types = self.filter_invalid_images()

    def __getitem__(self, index):
        pos_x, pos_y = self.tile_pos[index]
        print(f"Position X: {pos_x}, Position Y: {pos_y}")

        image = self.read_region(pos_x, pos_y, self.tile_size, self.tile_size)
        print("Image dimensions before transformation:", image.shape)

        if self.transform is not None:
            transformed_image = self.transform(image)
        else:
            transformed_image = transforms.ToTensor()(image)
        label = None
        x = pos_x
        y = pos_y
        img_id = index
        return transformed_image, label, x, y, img_id

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

    def load_tiles(self, tile_size):
        ''' Load tile positions from disk '''
        tile_path = f'{self.root_path}/tiles/positions_{tile_size}.csv'
        df = pd.read_csv(tile_path)
        tile_pos = df[["h", "w"]].to_numpy()
        self.cell_types = df["celltype"].values
        return tile_pos

    def filter_invalid_images(self):
        ''' Filter out invalid images and save valid cell types '''
        valid_tile_pos = []
        valid_cell_types = []

        for i, (pos_x, pos_y) in enumerate(self.tile_pos):
            try:
                image = self.read_region(pos_x, pos_y, self.tile_size, self.tile_size)
                if image.shape[0] > 0 and image.shape[1] > 0:
                    valid_tile_pos.append([pos_x, pos_y])
                    valid_cell_types.append(self.cell_types[i])
            except Exception as e:
                print(f"Skipping invalid image at position ({pos_x}, {pos_y}): {e}")

        # Save valid cell types to a NumPy file
        save_path = 'gpfs/scratch/ss14424/Brain/cells/analysis_output/celltype.npy'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, valid_cell_types)
        print(f"Cell types saved to {save_path}")

        # Update the tile positions to only include valid ones
        self.tile_pos = np.array(valid_tile_pos)
        return np.array(valid_cell_types)

    def load_tiling_mask(self, mask_path, tile_size):
        ''' Load tissue mask to generate tiles '''
        # Get slide dimensions
        slide_width, slide_height = self.get_slide_dimensions()
        # Specify grid size
        grid_width, grid_height = slide_width // tile_size, slide_height // tile_size
        # Create mask
        if mask_path is not None: # Load mask from existing file
            mask_temp = np.array(imread(mask_path)).swapaxes(0, 1)
            assert abs(mask_temp.shape[0] / mask_temp.shape[1] - slide_width / slide_height) < 0.01 , 'Mask shape does not match slide shape'
            # Convert mask to patch-pixel level grid
            mask = resize(mask_temp, (grid_width, grid_height), anti_aliasing=False)
        else:
            mask = np.ones(grid_width, grid_height) # Tile all regions
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
