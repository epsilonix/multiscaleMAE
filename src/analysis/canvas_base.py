#!/usr/bin/env python
"""
Integrated Canvas Base Script

This script combines the SCME and LTME versions into one. The user selects
the pipeline via the command-line argument --pipeline (choices: SCME or LTME).
"""

import os
import sys
import json
import torch
import numpy as np
import argparse
from tqdm import tqdm
from skimage.draw import polygon

def parse_args():
    parser = argparse.ArgumentParser("Canvas Embedding Extraction")
    parser.add_argument("--pipeline", type=str, default="SCME", choices=["SCME", "LTME"],
                        help="Select pipeline: SCME or LTME")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint (e.g. checkpoint-260.pth for SCME)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to image data (e.g. img_output_16_subsample for SCME or img_output_64 for LTME)")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Path where outputs will be saved")
    parser.add_argument("--tile_size", type=int, required=True,
                        help="Tile size (e.g. 16 for SCME, 64 for LTME)")
    return parser.parse_args()


def main():
    args = parse_args()
    canvas = Canvas(args.model_path, args.data_path, args.save_path, args.tile_size, pipeline=args.pipeline)
    
    # Load dataset and display first batch for sanity-check
    dataloader = canvas.load_dataset()
    first_batch = next(iter(dataloader))
    print("First Batch:", first_batch)
    
    # Load model and compute tile embeddings and clustering
    model = canvas.load_model(dataloader)
    canvas.get_tile_embedding(dataloader, model, save_full_emb=False)
    
    # Use different default number of clusters based on pipeline
    if canvas.pipeline == "SCME":
        canvas.clustering(n_clusters=50)
    else:
        canvas.clustering(n_clusters=40)
    

class Canvas:
    def __init__(self, model_path: str, data_path: str, save_path: str, tile_size, device: str = "cuda:0", pipeline: str = "SCME"):
        self.model_path = model_path
        self.data_path = data_path
        self.save_path = save_path
        self.tile_size = tile_size
        self.device = device
        self.pipeline = pipeline.upper()
        os.makedirs(save_path, exist_ok=True)
        self.step_dict = self.get_step_dict()

    def get_step_dict(self):
        step_dict_path = os.path.join(self.save_path, "step_dict.json")
        if os.path.exists(step_dict_path):
            with open(step_dict_path, "r") as f:
                step_dict = json.load(f)
        else:
            step_dict = {}
        print("Step_dict:", step_dict)
        return step_dict

    def flush_step_dict(self):
        with open(os.path.join(self.save_path, "step_dict.json"), "w") as f:
            json.dump(self.step_dict, f)

    def load_model(self, dataloader, norm_pix_loss=False, model_name="mae_vit_large_patch16"):
        num_channels = len(dataloader.dataset.common_channel_names)
        from model import models_mae
        model = models_mae.__dict__[model_name](norm_pix_loss=norm_pix_loss, in_chans=num_channels)
        model.to(self.device)
        print("Model initialized")
        state_dict = torch.load(self.model_path)["model"]
        model.load_state_dict(state_dict)
        print("State dicts loaded")
        model.eval()
        return model

    def load_dataset(self, batch_size=64, num_workers=40):
        from torchvision import transforms
        input_size = 224
        transform_codex = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((input_size, input_size), interpolation=2),
        ])
        # Conditionally import the appropriate dataset classes
        if self.pipeline == "SCME":
            from model.data.imc_dataset_scme import CANVASDatasetWithLocation, SlidesDataset
        else:
            from model.data.imc_dataset_ltme import CANVASDatasetWithLocation, SlidesDataset
        dataset = SlidesDataset(
            self.data_path,
            tile_size=self.tile_size,
            transform=transform_codex,
            dataset_class=CANVASDatasetWithLocation
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False,
        )
        return dataloader

    def get_tile_embedding(self, dataloader, model, output_suffix="tile_embedding", save_image=False, save_full_emb=False):
        output_path = os.path.join(self.save_path, output_suffix)
        os.makedirs(output_path, exist_ok=True)
        if output_suffix in self.step_dict and "embedding_mean" in self.step_dict[output_suffix]:
            if os.path.exists(self.step_dict[output_suffix]["embedding_mean"]):
                print("Embedding already exists, skipping")
                return
        data_size = len(dataloader.dataset)
        num_channels = len(dataloader.dataset.common_channel_names)
        embedding_shape = (196, 1024)
        if save_image:
            image_tensor = np.memmap(os.path.join(output_path, "image_tensor.dat"), dtype=np.float16, mode="w+", shape=(data_size, num_channels, 224, 224))
        if save_full_emb:
            embedding_tensor = np.memmap(os.path.join(output_path, "embedding_tensor.dat"), dtype=np.float16, mode="w+", shape=(data_size, *embedding_shape))
        image_mean_tensor = np.zeros((data_size, num_channels), dtype=np.float16)
        embedding_mean_tensor = np.zeros((data_size, 1024), dtype=np.float16)
        sample_name_list, tile_location_list = [], []
        celltype_list, boundary_list = [], []
        with torch.no_grad():
            for batch_idx, (img_tensor, extra) in enumerate(tqdm(dataloader)):
                data_idx = batch_idx * dataloader.batch_size
                temp_size = img_tensor.shape[0]
                embedding = self.proc_embedding(img_tensor, model)
                if self.pipeline == "SCME":
                    labels, locations, celltypes, boundaries = extra
                    sample_name_list.extend(labels)
                    tile_location_list.extend(locations)
                    celltype_list.extend(celltypes)
                    boundary_list.extend(boundaries)
                else:
                    labels, locations = extra
                    sample_name_list.extend(labels)
                    tile_location_list.extend(locations)
                image_mean_tensor[data_idx:data_idx+temp_size] = img_tensor.mean(dim=(2, 3)).to(torch.float16).cpu().numpy()
                embedding_mean_tensor[data_idx:data_idx+temp_size] = embedding.mean(dim=1).cpu().numpy().astype(np.float16)
                if save_image:
                    image_tensor[data_idx:data_idx+temp_size] = img_tensor.cpu().numpy().astype(np.float16)
                if save_full_emb:
                    embedding_tensor[data_idx:data_idx+temp_size] = embedding
        np.save(os.path.join(output_path, "image_mean.npy"), image_mean_tensor)
        np.save(os.path.join(output_path, "embedding_mean.npy"), embedding_mean_tensor)
        np.save(os.path.join(output_path, "tile_location.npy"), np.array(tile_location_list))
        np.save(os.path.join(output_path, "sample_name.npy"), np.array(sample_name_list))
        if self.pipeline == "SCME":
            np.save(os.path.join(output_path, "celltypes.npy"), np.array(celltype_list))
            np.save(os.path.join(output_path, "boundaries.npy"), np.array(boundary_list))
        if save_image:
            image_tensor.flush()
        if save_full_emb:
            embedding_tensor.flush()
        tile_dict = {
            "image_mean": os.path.join(output_path, "image_mean.npy"),
            "embedding_mean": os.path.join(output_path, "embedding_mean.npy"),
            "tile_location": os.path.join(output_path, "tile_location.npy"),
            "sample_name": os.path.join(output_path, "sample_name.npy")
        }
        if self.pipeline == "SCME":
            tile_dict["celltypes"] = os.path.join(output_path, "celltypes.npy")
            tile_dict["boundaries"] = os.path.join(output_path, "boundaries.npy")
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

    def clustering(self, output_suffix="clustering", n_clusters=None):
        if n_clusters is None:
            n_clusters = 50 if self.pipeline == "SCME" else 40
        output_path = os.path.join(self.save_path, output_suffix)
        os.makedirs(output_path, exist_ok=True)
        save_path = os.path.join(output_path, "labels.npy")
        from analysis.clustering import kmeans
        kmeans.clustering(self.step_dict["tile_embedding"]["embedding_mean"], n_clusters, save_path)
        data_dict = {"labels": save_path}
        self.step_dict[output_suffix] = data_dict
        self.flush_step_dict()


if __name__ == '__main__':
    main()
