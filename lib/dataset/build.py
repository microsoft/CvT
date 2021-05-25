from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

from timm.data import create_loader
import torch
import torch.utils.data
import torchvision.datasets as datasets

from .transformas import build_transforms
from .samplers import RASampler


def build_dataset(cfg, is_train):
    dataset = None
    if 'imagenet' in cfg.DATASET.DATASET:
        dataset = _build_imagenet_dataset(cfg, is_train)
    else:
        raise ValueError('Unkown dataset: {}'.format(cfg.DATASET.DATASET))
    return dataset


def _build_image_folder_dataset(cfg, is_train):
    transforms = build_transforms(cfg, is_train)

    dataset_name = cfg.DATASET.TRAIN_SET if is_train else cfg.DATASET.TEST_SET
    dataset = datasets.ImageFolder(
        os.path.join(cfg.DATASET.ROOT, dataset_name), transforms
    )
    logging.info(
        '=> load samples: {}, is_train: {}'
        .format(len(dataset), is_train)
    )

    return dataset


def _build_imagenet_dataset(cfg, is_train):
    transforms = build_transforms(cfg, is_train)

    dataset_name = cfg.DATASET.TRAIN_SET if is_train else cfg.DATASET.TEST_SET
    dataset = datasets.ImageFolder(
        os.path.join(cfg.DATASET.ROOT, dataset_name), transforms
    )

    return dataset


def build_dataloader(cfg, is_train=True, distributed=False):
    if is_train:
        batch_size_per_gpu = cfg.TRAIN.BATCH_SIZE_PER_GPU
        shuffle = True
    else:
        batch_size_per_gpu = cfg.TEST.BATCH_SIZE_PER_GPU
        shuffle = False

    dataset = build_dataset(cfg, is_train)

    if distributed:
        if is_train and cfg.DATASET.SAMPLER == 'repeated_aug':
            logging.info('=> use repeated aug sampler')
            sampler = RASampler(dataset, shuffle=shuffle)
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=shuffle
            )
        shuffle = False
    else:
        sampler = None

    if cfg.AUG.TIMM_AUG.USE_LOADER and is_train:
        logging.info('=> use timm loader for training')
        timm_cfg = cfg.AUG.TIMM_AUG
        data_loader = create_loader(
            dataset,
            input_size=cfg.TRAIN.IMAGE_SIZE[0],
            batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU,
            is_training=True,
            use_prefetcher=True,
            no_aug=False,
            re_prob=timm_cfg.RE_PROB,
            re_mode=timm_cfg.RE_MODE,
            re_count=timm_cfg.RE_COUNT,
            re_split=timm_cfg.RE_SPLIT,
            scale=cfg.AUG.SCALE,
            ratio=cfg.AUG.RATIO,
            hflip=timm_cfg.HFLIP,
            vflip=timm_cfg.VFLIP,
            color_jitter=timm_cfg.COLOR_JITTER,
            auto_augment=timm_cfg.AUTO_AUGMENT,
            num_aug_splits=0,
            interpolation=timm_cfg.INTERPOLATION,
            mean=cfg.INPUT.MEAN,
            std=cfg.INPUT.STD,
            num_workers=cfg.WORKERS,
            distributed=distributed,
            collate_fn=None,
            pin_memory=cfg.PIN_MEMORY,
            use_multi_epochs_loader=True
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size_per_gpu,
            shuffle=shuffle,
            num_workers=cfg.WORKERS,
            pin_memory=cfg.PIN_MEMORY,
            sampler=sampler,
            drop_last=True if is_train else False,
        )

    return data_loader
