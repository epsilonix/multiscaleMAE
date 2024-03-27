# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import model.util.misc as misc
import model.util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True, dtype=torch.float32)

        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
#        if (data_iter_step + 1) % accum_iter == 0:
#            optimizer.zero_grad()
        # Gradient logging goes here
        # Initialize a dictionary to store the sum of gradient norms and counts
        grad_norm_sum = {name: 0.0 for name, _ in model.named_parameters()}
        grad_norm_count = {name: 0 for name, _ in model.named_parameters()}

        # ... Inside your training loop, accumulate the norms
        if (data_iter_step + 1) % accum_iter == 0:
            for name, parameter in model.named_parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    grad_norm_sum[name] += parameter.grad.norm().item()
                    grad_norm_count[name] += 1

        # ... After the training loop (at the end of an epoch), log the average norms
        if log_writer is not None:
            for name in grad_norm_sum:
                if grad_norm_count[name] > 0:
                    average_grad_norm = grad_norm_sum[name] / grad_norm_count[name]
                    log_writer.add_scalar(f'grad_norm_avg/{name}', average_grad_norm, epoch)
            # Reset for the next epoch
            grad_norm_sum = {name: 0.0 for name in grad_norm_sum}
            grad_norm_count = {name: 0 for name in grad_norm_count}

    
    

        optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
