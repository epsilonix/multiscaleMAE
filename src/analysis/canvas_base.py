import os
import json
import psutil  # Add this import for memory profiling
import torch
import numpy as np
import pandas as pd

import sys
sys.path.append('/gpfs/scratch/ss14424/singlecell/src')

from tqdm import tqdm

# Memory check function
def check_memory_usage(step_name):
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"[{step_name}] Memory Usage: {memory_info.rss / (1024 ** 3):.2f} GB")  # rss is the resident set size

def main():
    check_memory_usage('Start')  # Check memory usage at the start
    # Initialize CANVAS
    data_path = '/gpfs/scratch/ss14424/Brain/channels_37/cells_blankout/img_output_16_subsample'
    model_path = '/gpfs/scratch/ss14424/Brain/channels_37/cells_blankout/model_output_20/checkpoint-160.pth'
    save_path = '/gpfs/scratch/ss14424/Brain/channels_37/cells_blankout/analysis_output_subsample/'
    tile_size = 16
    canvas = Canvas(model_path, data_path, save_path, tile_size)
    check_memory_usage('Initialized Canvas')  # Check after initializing Canvas

    # Generate embeddings
    dataloader = canvas.load_dataset()
    check_memory_usage('Data Loader Created')  # Check after data loader creation

    first_batch = next(iter(dataloader))
    print("First Batch:", first_batch)
    check_memory_usage('First Batch Loaded')  # Check after loading the first batch

    model = canvas.load_model(dataloader)
    check_memory_usage('Model Loaded')  # Check after loading the model

    canvas.get_tile_embedding(dataloader, model, save_full_emb=False)
    check_memory_usage('Tile Embedding Completed')  # Check after generating embeddings

    canvas.clustering(n_clusters=60)
    check_memory_usage('Clustering Completed')  # Check after clustering
    
class Canvas:

    def __init__(self, model_path : str, data_path : str, save_path : str,
                 tile_size, 
                 device : str = 'cuda:0') -> None:
        self.model_path = model_path
        self.data_path = data_path
        self.save_path = save_path
        self.tile_size = tile_size
        self.device = device
        os.makedirs(save_path, exist_ok = True)
        self.step_dict = self.get_step_dict()

    def get_step_dict(self):
        step_dict_save_path = f'{self.save_path}/step_dict.json'
        if os.path.exists(step_dict_save_path):
            step_dict = json.load(open(step_dict_save_path, 'r'))
        else:
            step_dict = {}
        print(f'Step_dict: {step_dict}')
        return step_dict
    

    def flush_step_dict(self):
        json.dump(self.step_dict, open(f'{self.save_path}/step_dict.json', 'w'))

    def load_model(self, dataloader, 
                   norm_pix_loss = False, model_name = 'mae_vit_large_patch16'):
        num_channels = len(dataloader.dataset.common_channel_names)
        from model import models_mae
        model = models_mae.__dict__[model_name](norm_pix_loss=norm_pix_loss, 
                                                in_chans = num_channels)
        model.to(self.device)
        print('Model initialized')
        state_dict = torch.load(self.model_path)['model']
        model.load_state_dict(state_dict)
        print('State dicts loaded')
        model.eval()
        return model 


    def load_dataset(self, batch_size = 128, num_workers = 40):
        # Predefined parameters
        input_size = 224
        from torchvision import transforms
        transform_codex = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((input_size,input_size), interpolation = 2),
                ])

        from model.data.imc_dataset import CANVASDatasetWithLocation, SlidesDataset
        dataset = SlidesDataset(self.data_path, tile_size = self.tile_size, transform = transform_codex, dataset_class = CANVASDatasetWithLocation)

        dataloader= torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False,
        )
        return dataloader


    def get_tile_embedding(self, dataloader, model, output_suffix='tile_embedding', save_image=False, save_full_emb=False):
        output_path = f'{self.save_path}/{output_suffix}'
        os.makedirs(output_path, exist_ok=True)

        if output_suffix in self.step_dict and 'embedding_mean' in self.step_dict[output_suffix]:
            if os.path.exists(self.step_dict[output_suffix]['embedding_mean']):
                print('Embedding already exists, skipping')
                return

        # Setup tensors and lists for storage
        data_size = len(dataloader.dataset)
        num_channels = len(dataloader.dataset.common_channel_names)
        embedding_shape = (196, 1024)  # Explicitly defining embedding shape

        # Use memory mapping for large arrays
        if save_image:
            image_tensor = np.memmap(f'{output_path}/image_tensor.dat', dtype=np.float16, mode='w+', shape=(data_size, num_channels, 224, 224))
        if save_full_emb:
            embedding_tensor = np.memmap(f'{output_path}/embedding_tensor.dat', dtype=np.float16, mode='w+', shape=(data_size, *embedding_shape))

        image_mean_tensor = np.zeros((data_size, num_channels), dtype=np.float16)
        embedding_mean_tensor = np.zeros((data_size, 1024), dtype=np.float16)
        sample_name_list, tile_location_list, celltype_list, boundary_list = [], [], [], []

        # Data processing and extraction
        with torch.no_grad():
            for batch_idx, (img_tensor, (labels, locations, celltypes, boundaries)) in enumerate(tqdm(dataloader)):
                data_idx = batch_idx * dataloader.batch_size
                temp_size = img_tensor.shape[0]
                embedding = self.proc_embedding(img_tensor, model)

                sample_name_list.extend(labels)
                tile_location_list.extend(locations)
                celltype_list.extend(celltypes)
                boundary_list.extend(boundaries)

                image_mean_tensor[data_idx:data_idx + temp_size] = img_tensor.mean(axis=(2, 3)).to(torch.float16).cpu().numpy()
                embedding_mean_tensor[data_idx:data_idx + temp_size] = embedding.mean(axis=1).astype(np.float16)
                if save_image:
                    image_tensor[data_idx:data_idx + temp_size] = img_tensor.numpy().astype(np.float16)
                if save_full_emb:
                    embedding_tensor[data_idx:data_idx + temp_size] = embedding

        # Save tensors to disk
        np.save(os.path.join(output_path, 'image_mean.npy'), image_mean_tensor)
        np.save(os.path.join(output_path, 'embedding_mean.npy'), embedding_mean_tensor)
        np.save(os.path.join(output_path, 'tile_location.npy'), np.array(tile_location_list))
        np.save(os.path.join(output_path, 'sample_name.npy'), np.array(sample_name_list))
        np.save(os.path.join(output_path, 'celltypes.npy'), np.array(celltype_list))
        np.save(os.path.join(output_path, 'boundaries.npy'), np.array(boundary_list))

        if save_image:
            image_tensor.flush()  # Ensure data is written to disk
        if save_full_emb:
            embedding_tensor.flush()  # Ensure data is written to disk

        # Update the step dictionary
        tile_dict = {
            'image_mean': os.path.join(output_path, 'image_mean.npy'),
            'embedding_mean': os.path.join(output_path, 'embedding_mean.npy'),
            'tile_location': os.path.join(output_path, 'tile_location.npy'),
            'sample_name': os.path.join(output_path, 'sample_name.npy'),
            'celltypes': os.path.join(output_path, 'celltypes.npy'),
            'boundaries': os.path.join(output_path, 'boundaries.npy')
        }
        self.step_dict[output_suffix] = tile_dict
        self.flush_step_dict()

      
    def proc_embedding(self, img_tensor, model):
        imgs = img_tensor.to(self.device).float()
        mask_ratio = 0
        with torch.no_grad():
            latent, mask, ids_restore = model.forward_encoder(imgs, mask_ratio)
            latent_no_cls = latent[:, 1:, :]
            restored_latent = torch.gather(latent_no_cls, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, latent.shape[2])).detach().cpu().numpy().astype(np.float16)
        return restored_latent

    def get_umap(self, output_suffix = 'umap'):
        output_path = f'{self.save_path}/{output_suffix}'
        os.makedirs(output_path, exist_ok = True)
        if output_suffix in self.step_dict and 'coord' in self.step_dict[output_suffix]:
            if os.path.exists(self.step_dict[output_suffix]['coord']):
                print('UMAP coord already exist, skipping')
                return 
        from analysis.umap_reduction.gen_umap_embedding import plot_umap
        umap_file = os.path.join(output_path, 'coord.npy')
        plot_umap(self.step_dict['tile_embedding']['embedding_mean'], umap_file)
        umap_dict = {'coord' : os.path.join(output_path, umap_file)}
        self.step_dict[output_suffix] = umap_dict
        self.flush_step_dict()

    def visualize_image(self, dataloader, color_map, image):
        color_pallete = {
            'blue': np.array([0, 0, 255]),
            'red': np.array([255, 0, 0]),
            'green': np.array([0, 255, 0]),
            'yellow': np.array([255, 255, 0]),
            'magenta': np.array([255, 0, 255]),
            'orange': np.array([255, 127, 0]),
            'white': np.array([255, 255, 255]),
            'cyan': np.array([0, 255, 255]),
            'lime': np.array([50, 205, 50]),  # Approximation for lime
            'pink': np.array([255, 192, 203]),
            'grey': np.array([128, 128, 128]),
            'olive': np.array([128, 128, 0]),
            'brown': np.array([165, 42, 42]),
            'navy': np.array([0, 0, 128]),
            'teal': np.array([0, 128, 128]),
            'maroon': np.array([128, 0, 0]),
            'purple': np.array([128, 0, 128]),
            'gold': np.array([255, 215, 0]),
            'lavender': np.array([230, 230, 250]),
            'skyblue': np.array([0, 191, 255]), 
            'violet': np.array([148, 0, 211])
        }
        channel_names = dataloader.dataset.common_channel_names
        print(f'channel names: {channel_names}')
        name_map = dict(zip(channel_names, range(len(channel_names))))
        print(f'name_map: {name_map}')
        channel_weights = dict(zip(channel_names, np.ones(len(channel_names))))
        print(f'channel_weights: {channel_weights}')
        
        #breakpoint()
        #image = image.swapaxes(0, 1).swapaxes(1, 2)

        rgb_image = np.zeros_like(image)[:, :, :3]
        print(f'rgb_image:{rgb_image}')
        for c_name, marker in color_map.items():
            print(f'c_name:{c_name}, marker:{marker}')
            print(f'name_map[marker]: {name_map[marker]}')
            marker_map_1d = image[:, :, name_map[marker]]
            marker_map_1d *= channel_weights[marker]
            marker_map = np.moveaxis(np.tile(marker_map_1d, (3, 1, 1)), 0, 2)
            final_map = (marker_map / 255) * color_pallete[c_name].reshape(1, 1, 3)
            rgb_image = np.maximum(rgb_image, final_map)
        return (rgb_image * 20).clip(0, 255).astype(np.uint8)

    def get_umap_mosaic(self, dataloader, color_map):
        # Does not save intermediate data
        output_path = f'{self.save_path}/umap'
        os.makedirs(output_path, exist_ok = True)
        output_file = os.path.join(output_path, 'umap_mosaic.png')
        if 'coord' not in self.step_dict['umap']:
            print('UMAP coord not found, skipping')
            return
        vis = lambda x : self.visualize_image(dataloader, color_map, x)
        from analysis.umap_reduction.gen_umap_mosaic import plot_umap_mosaic
        plot_umap_mosaic(self.step_dict['umap']['coord'], output_file, dataloader, vis)

    def color_umap(self, dataloader):
        # Does not save intermediate data
        output_path = f'{self.save_path}/umap'
        os.makedirs(output_path, exist_ok = True)
        if 'coord' not in self.step_dict['umap']:
            print('UMAP coord not found, skipping')
            return

        # Color by marker
        channel_names = dataloader.dataset.common_channel_names
        from analysis.umap_reduction import color_by_marker
        
        
        color_by_marker.plot(self.step_dict['umap']['coord'], self.step_dict['tile_embedding']['image_mean'], f'{output_path}/umap_by_marker.png', channel_names)

        # Color by sample name
        from analysis.umap_reduction import color_by_sample
        color_by_sample.plot(self.step_dict['umap']['coord'], self.step_dict['tile_embedding']['sample_name'], f'{output_path}/umap_by_sample.png')

    def clustering(self, output_suffix = 'clustering', n_clusters = 20):
        output_path = f'{self.save_path}/{output_suffix}'
        os.makedirs(output_path, exist_ok = True)
        save_path = os.path.join(output_path, 'labels.npy')
#        if output_suffix in self.step_dict and 'labels' in self.step_dict[output_suffix]:
#            if os.path.exists(self.step_dict[output_suffix]['labels']):
#                print('Embedding already exist, skipping')
#                return 
        from analysis.clustering import kmeans
        kmeans.clustering(self.step_dict['tile_embedding']['embedding_mean'], n_clusters, save_path)
        data_dict = {'labels' : save_path}
        self.step_dict[output_suffix] = data_dict
        self.flush_step_dict()
    

if __name__ == '__main__':
    main()
