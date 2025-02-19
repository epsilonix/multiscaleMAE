import numpy as np
import pandas as pd
import os
import torch.utils.data as data
from torchvision import transforms
from skimage.io import imsave, imread
from skimage.transform import resize
from skimage.draw import polygon
import json
import matplotlib.pyplot as plt

class SlideDataset(data.Dataset):
    """Dataset for slides."""
    def __init__(self, root_path=None, tile_size=None, transform=None):
        """
        Initialize the dataset.
        root_path: root path of the dataset (for saving processed files)
        """
        self.root_path = root_path
        self.tile_size = tile_size
        self.transform = transform

        self.df = self.load_tile_data(tile_size)
        self.tile_pos = self.load_tiles()
        self.celltypes = self.load_celltypes()
        self.boundary = self.load_boundary()

    def __getitem__(self, index):
        # Load the image tile based on its top-left coordinates
        x, y = self.tile_pos[index][0], self.tile_pos[index][1]
        image = self.read_region(x, y, self.tile_size, self.tile_size)
        if self.transform is not None:
            transformed_image = self.transform(image)
        else:
            transformed_image = transforms.ToTensor()(image)
        label = None
        img_id = index
        return transformed_image, label, x, y, img_id

    def __len__(self):
        return len(self.tile_pos)

    def read_slide(self, root_path):
        """Read slide from disk."""
        raise NotImplementedError

    def read_region(self, pos_x, pos_y, width, height):
        """Return region of the slide given top-left corner coordinates."""
        raise NotImplementedError

    def get_slide_dimensions(self):
        """Return slide dimensions."""
        raise NotImplementedError

    def save_thumbnail(self):
        """Save a thumbnail of the slide."""
        raise NotImplementedError

    def load_tile_data(self, tile_size):
        tile_path = f'{self.root_path}/tiles/positions_{tile_size}.csv'
        df = pd.read_csv(tile_path)
        return df

    def load_tiles(self):
        tile_pos = self.df[["h", "w"]].to_numpy()
        return tile_pos

    def load_celltypes(self):
        celltypes = self.df["celltype"].values
        return celltypes

    def load_boundary(self):
        boundary = self.df["boundary"].values
        return boundary

    def load_tiling_mask(self, mask_path, tile_size):
        slide_width, slide_height = self.get_slide_dimensions()
        grid_width, grid_height = slide_width // tile_size, slide_height // tile_size
        if mask_path is not None:
            mask_temp = np.array(imread(mask_path)).swapaxes(0, 1)
            assert abs(mask_temp.shape[0] / mask_temp.shape[1] - slide_width / slide_height) < 0.01, 'Mask shape does not match slide shape'
            mask = resize(mask_temp, (grid_width, grid_height), anti_aliasing=False)
        else:
            mask = np.ones((grid_width, grid_height))
        return mask

    def generate_tiles(self, tile_size, mask_path=None, mask_id='default', threshold=0.99):
        mask = self.load_tiling_mask(mask_path, tile_size)
        ws, hs = np.where(mask >= threshold)
        positions = (np.array(list(zip(ws, hs))) * tile_size)
        tile_path = f'{self.root_path}/tiles/{mask_id}'
        save_path = f'{tile_path}/{tile_size}'
        os.makedirs(save_path, exist_ok=True)
        np.save(f'{save_path}/tile_positions_top_left.npy', positions)
        mask_img = np.zeros_like(mask)
        mask_img[ws, hs] = 1
        imsave(f'{save_path}/mask.png', (mask_img.swapaxes(0, 1) * 255).astype(np.uint8))
