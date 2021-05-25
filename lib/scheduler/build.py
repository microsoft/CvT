from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from timm.scheduler import create_scheduler


def build_lr_scheduler(cfg, optimizer, begin_epoch):
    if 'METHOD' not in cfg.TRAIN.LR_SCHEDULER:
        raise ValueError('Please set TRAIN.LR_SCHEDULER.METHOD!')
    elif cfg.TRAIN.LR_SCHEDULER.METHOD == 'MultiStep':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            cfg.TRAIN.LR_SCHEDULER.MILESTONES,
            cfg.TRAIN.LR_SCHEDULER.GAMMA,
            begin_epoch - 1)
    elif cfg.TRAIN.LR_SCHEDULER.METHOD == 'CosineAnnealing':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            cfg.TRAIN.END_EPOCH,
            cfg.TRAIN.LR_SCHEDULER.ETA_MIN,
            begin_epoch - 1
        )
    elif cfg.TRAIN.LR_SCHEDULER.METHOD == 'CyclicLR':
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=cfg.TRAIN.LR_SCHEDULER.BASE_LR,
            max_LR=cfg.TRAIN.LR_SCHEDULER.MAX_LR,
            step_size_up=cfg.TRAIN.LR_SCHEDULER.STEP_SIZE_UP
        )
    elif cfg.TRAIN.LR_SCHEDULER.METHOD == 'timm':
        args = cfg.TRAIN.LR_SCHEDULER.ARGS
        lr_scheduler, _ = create_scheduler(args, optimizer)
        lr_scheduler.step(begin_epoch)
    else:
        raise ValueError('Unknown lr scheduler: {}'.format(
            cfg.TRAIN.LR_SCHEDULER.METHOD))

    return lr_scheduler

