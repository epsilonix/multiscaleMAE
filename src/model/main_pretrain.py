#!/usr/bin/env python
"""
This source code is modified from the following repositories:
--------------------------------------------------------
References:
DeiT: https://github.com/facebookresearch/deit
BEiT: https://github.com/microsoft/unilm/tree/master/beit
MAE: https://github.com/facebookresearch/mae
--------------------------------------------------------
"""

import traceback
import argparse
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
import torchvision

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import model.util.misc as misc
from model.util.misc import NativeScalerWithGradNormCount as NativeScaler

from model import models_mae
from model.engine_pretrain import train_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus)')
    parser.add_argument('--epochs', default=1400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='', type=str,
                        help='dataset path')
    parser.add_argument('--tile_size', default=16, type=int,
                        help='Sample tile size.')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    # Pipeline selection argument
    parser.add_argument('--pipeline', type=str, default='SCME', choices=['SCME', 'LTME'],
                        help='Pipeline to use: SCME or LTME')
    return parser


def channel_augment(image):
    num_channels = image.shape[0]
    augmented_image = image * torch.exp(
        torch.normal(torch.zeros(num_channels), torch.ones(num_channels) * 0.3)
            .unsqueeze(1).unsqueeze(2))
    return augmented_image


def main(args):
    misc.init_distributed_mode(args)
    print('Job directory: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("Arguments:\n{}".format(args).replace(', ', ',\n'))
    print(f"Using pipeline: {args.pipeline.upper()}")

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    try:
        transform_codex = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(args.input_size, interpolation=2),
            transforms.RandomRotation(degrees=180, interpolation=2, expand=True),
            transforms.CenterCrop(args.input_size),
            transforms.RandomResizedCrop(args.input_size, scale=(0.8, 1.0), ratio=(1, 1), interpolation=2),
            transforms.RandomHorizontalFlip(),
            channel_augment
        ])

        # Conditionally import dataset classes based on the chosen pipeline
        if args.pipeline.upper() == 'SCME':
            from data.imc_dataset_scme import CANVASDataset, SlidesDataset
        else:
            from data.imc_dataset_ltme import CANVASDataset, SlidesDataset

        dataset_train = SlidesDataset(
            args.data_path,
            tile_size=args.tile_size,
            transform=transform_codex,
            dataset_class=CANVASDataset
        )

        sample_tile, _ = dataset_train[0]
        print(f"Sample tile size: {sample_tile.shape}")

        if args.distributed:
            num_tasks = misc.get_world_size()
            global_rank = misc.get_rank()
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            print("Sampler_train = %s" % str(sampler_train))
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)

        log_writer = None
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

        num_channels = len(dataset_train.common_channel_names)
        print(f'Number of channels: {num_channels}')
        model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, in_chans=num_channels)
        model.to(device)
        model_without_ddp = model
        print("Model = %s" % str(model_without_ddp))

        eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
        if args.lr is None:
            args.lr = args.blr * eff_batch_size / 256

        print("Base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        print("Actual lr: %.2e" % args.lr)
        print("Accumulate grad iterations: %d" % args.accum_iter)
        print("Effective batch size: %d" % eff_batch_size)

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module

        param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
        print(optimizer)
        loss_scaler = NativeScaler()
        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()

        try:
            for epoch in range(args.start_epoch, args.epochs):
                if args.distributed:
                    data_loader_train.sampler.set_epoch(epoch)
                train_stats = train_one_epoch(
                    model, data_loader_train,
                    optimizer, device, epoch, loss_scaler,
                    log_writer=log_writer,
                    args=args
                )
                if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
                    misc.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch)
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             'epoch': epoch,}
                if args.output_dir and misc.is_main_process():
                    with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                        f.write(json.dumps(log_stats) + "\n")
        except Exception as e:
            print("An error occurred during training:")
            print(e)
            print(traceback.format_exc())

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    except Exception as e:
        print("An error occurred during setup:")
        print(e)
        print(traceback.format_exc())


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    main(args)
