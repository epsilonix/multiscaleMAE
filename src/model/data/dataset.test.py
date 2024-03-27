import os
import torch
import numpy as np
from torchvision import transforms
from data.imc_dataset import CANVASDataset, SlidesDataset

def main():
    root_path = '/gpfs/scratch/jt3545/projects/CODEX/analysis/kidney/data'
    tile_size = 128


    transform_codex = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224, interpolation = 2),
            transforms.RandomRotation(degrees = 180, interpolation = 2, expand = True),
            transforms.CenterCrop(224 * 0.7), 
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(1, 1), interpolation=2), # 2 is bilinear
            transforms.RandomHorizontalFlip(),
            channel_augment,
            ])

    dataset = SlidesDataset(root_path, tile_size, transform_codex, CANVASDataset)

    dataset[0]
    random_idxs = np.random.randint(0, len(dataset), 10)

    for i in random_idxs:
        image = dataset[i][0]
        print(image.shape)
        breakpoint()
        '''
        image = (image - image.min()) / (image.max() - image.min()) * 255
        rgb_image = vis_codex(image.astype(np.float64))
        from skimage.io import imsave
        os.makedirs('test', exist_ok = True)
        imsave(f'test/test_{i}.png', rgb_image)
        '''

def channel_augment(image):
    num_channels = image.shape[0]
    augmented_image = image * torch.exp(torch.normal(torch.zeros(num_channels), torch.ones(num_channels) * 0.3).unsqueeze(1).unsqueeze(2))
    return augmented_image

def vis_codex(image):

    color_pallete = {'blue' : np.array([0, 0, 255]),
                     'green' : np.array([0, 255, 0]),
                     'yellow' : np.array([255, 255, 0]),
                     'magenta' : np.array([255, 0, 255]),
                     'orange' : np.array([255, 127, 0]),
                     'white' : np.array([255, 255, 255]),
                     }
    channel_names = ["CD117", "CD11c", "CD14", "CD163", "CD16", "CD20", "CD31", "CD3", "CD4", "CD68", "CD8a", "CD94", "DNA1", "FoxP3", "HLA-DR", "MPO", "Pancytokeratin", "TTF1"]
    name_map = dict(zip(channel_names, range(len(channel_names))))
    color_map = {'blue' : 'DNA1',
                 'green' : 'CD20',
                 'yellow' : 'HLA-DR',
                 'magenta' : 'Pancytokeratin',
                 'orange' : 'MPO',
                 }

    channel_weights = dict(zip(channel_names, np.ones(len(channel_names))))
    channel_weights['CD8a'] = 1
    channel_weights['DNA1'] = 1
    channel_weights['HLA-DR'] = 1
    channel_weights['Pancytokeratin'] = 1

    image = image.swapaxes(0, 1).swapaxes(1, 2)
    rgb_image = np.zeros_like(image)[:, :, :3]
    for c_name, marker in color_map.items():
        marker_map_1d = image[:, :, name_map[marker]]
        marker_map_1d *= channel_weights[marker]
        marker_map = np.moveaxis(np.tile(marker_map_1d, (3, 1, 1)), 0, 2)
        final_map = (marker_map / 255) * color_pallete[c_name].reshape(1, 1, 3)
        rgb_image = np.maximum(rgb_image, final_map)
    return (rgb_image * 2).clip(0, 255).astype(np.uint8)

if __name__ == '__main__':
    main()
