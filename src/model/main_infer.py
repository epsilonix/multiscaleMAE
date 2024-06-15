# This rce code is modified from the following repositories:
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import sys
sys.path.append('/gpfs/scratch/ss14424/CANVAS-ss/singlecell')

import argparse
import matplotlib.pyplot as plt
from PIL import Image
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae


def get_args_parser():
    parser = argparse.ArgumentParser('MAE inference', add_help=False)
    # Model parameters
    parser.add_argument('--chkpt_dir', default='/gpfs/scratch/ss14424/Brain/cells_csd/model_output_20/checkpoint-300.pth', type=str, metavar='CHKPT',
                        help='Checkpoint path')

    # Dataset parameters
    parser.add_argument('--data_path', default='/gpfs/scratch/ss14424/Brain/cells_csd/img_output_10', type=str,
                        help='dataset path')
    parser.add_argument('--tile_size', default=10, type=int,
                        help='Sample tile size.')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--output_dir', default='/gpfs/scratch/ss14424/Brain/cells_csd/reconstruction',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/gpfs/scratch/ss14424/logs',
                        help='path where to tensorboard log')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    

    return parser

# def channel_augment(image):
#     num_channels = image.shape[0]
#     augmented_image = image * torch.exp(torch.normal(torch.zeros(num_channels), torch.ones(num_channels) * 0.3).unsqueeze(1).unsqueeze(2))
#     return augmented_image

                        
def show_image(image, title='', dapi_index = 0):
    print("image shape")
    print(np.shape(image))
    # Get DAPI
    image = image[dapi_index,:,:]
#     image = np.transpose(image, (1, 2, 0))
                        
    # image is [H, W, 3]
#     assert image.shape[2] == 1
    plt.imshow(image)
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def prepare_model(chkpt_dir, norm_pix_loss, num_channels, arch='mae_vit_large_patch16'):
    # build model
    model = models_mae.__dict__[arch](norm_pix_loss=norm_pix_loss, in_chans = num_channels)
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def run_one_image(img, model, output_dir, nb_sample, num_channels, dapi_index):
    x = torch.tensor(img)
    x = x.unsqueeze(dim=0)

    # run MAE
    loss, y, mask = model(x.float(), mask_ratio=0.75)
    y = model.unpatchify(y)
    y = y.detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *num_channels)  # (N, H*W, p*p*39)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = mask.detach().cpu()
    
    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [16, 4]

    plt.subplot(1, 4, 1)
    show_image(x[0], "original", dapi_index)

    plt.subplot(1, 4, 2)
    show_image(im_masked[0], "masked", dapi_index)

    plt.subplot(1, 4, 3)
    show_image(y[0], "reconstruction", dapi_index)

    plt.subplot(1, 4, 4)
    show_image(im_paste[0], "reconstruction + visible", dapi_index)

    plt.savefig(os.path.join(output_dir,"example_" + str(nb_sample) + ".png"))


def main(args):

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    '''
    with open(f'{args.data_path}/common_channels.txt', 'r') as f:
        channel_names = f.read().splitlines()
    num_channels = len(channel_names)
    '''
    
    transform_codex = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(args.input_size, interpolation = 2),
        ])

    from data.imc_dataset import CANVASDataset, SlidesDataset
    tile_size = args.tile_size
    dataset_train = SlidesDataset(args.data_path, tile_size = tile_size, transform = transform_codex, dataset_class = CANVASDataset)
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
                           
    # Load model
    num_channels = len(dataset_train.common_channel_names)
    print(f'Number of channels: {num_channels}')
    dapi_index = dataset_train.common_channel_names.index("DNA1")
    model = prepare_model(args.chkpt_dir, args.norm_pix_loss, num_channels, 'mae_vit_large_patch16')                    
    
    nb_sample = 0

    with torch.no_grad():
        for batch_idx, (samples, _) in enumerate(data_loader_train):
            for sample in samples:
                img = sample.numpy()
                run_one_image(img, model, args.output_dir, nb_sample, num_channels, dapi_index)
                nb_sample += 1

         
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    main(args)