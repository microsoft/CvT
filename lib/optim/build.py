from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.optim as optim

from timm.optim import create_optimizer


def _is_depthwise(m):
    return (
        isinstance(m, nn.Conv2d)
        and m.groups == m.in_channels
        and m.groups == m.out_channels
    )


def set_wd(cfg, model):
    without_decay_list = cfg.TRAIN.WITHOUT_WD_LIST
    without_decay_depthwise = []
    without_decay_norm = []
    for m in model.modules():
        if _is_depthwise(m) and 'dw' in without_decay_list:
            without_decay_depthwise.append(m.weight)
        elif isinstance(m, nn.BatchNorm2d) and 'bn' in without_decay_list:
            without_decay_norm.append(m.weight)
            without_decay_norm.append(m.bias)
        elif isinstance(m, nn.GroupNorm) and 'gn' in without_decay_list:
            without_decay_norm.append(m.weight)
            without_decay_norm.append(m.bias)
        elif isinstance(m, nn.LayerNorm) and 'ln' in without_decay_list:
            without_decay_norm.append(m.weight)
            without_decay_norm.append(m.bias)

    with_decay = []
    without_decay = []

    skip = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()

    skip_keys = {}
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keys = model.no_weight_decay_keywords()

    for n, p in model.named_parameters():
        ever_set = False

        if p.requires_grad is False:
            continue

        skip_flag = False
        if n in skip:
            print('=> set {} wd to 0'.format(n))
            without_decay.append(p)
            skip_flag = True
        else:
            for i in skip:
                if i in n:
                    print('=> set {} wd to 0'.format(n))
                    without_decay.append(p)
                    skip_flag = True

        if skip_flag:
            continue

        for i in skip_keys:
            if i in n:
                print('=> set {} wd to 0'.format(n))

        if skip_flag:
            continue

        for pp in without_decay_depthwise:
            if p is pp:
                if cfg.VERBOSE:
                    print('=> set depthwise({}) wd to 0'.format(n))
                without_decay.append(p)
                ever_set = True
                break

        for pp in without_decay_norm:
            if p is pp:
                if cfg.VERBOSE:
                    print('=> set norm({}) wd to 0'.format(n))
                without_decay.append(p)
                ever_set = True
                break

        if (
            (not ever_set)
            and 'bias' in without_decay_list
            and n.endswith('.bias')
        ):
            if cfg.VERBOSE:
                print('=> set bias({}) wd to 0'.format(n))
            without_decay.append(p)
        elif not ever_set:
            with_decay.append(p)

    # assert (len(with_decay) + len(without_decay) == len(list(model.parameters())))
    params = [
        {'params': with_decay},
        {'params': without_decay, 'weight_decay': 0.}
    ]
    return params


def build_optimizer(cfg, model):
    if cfg.TRAIN.OPTIMIZER == 'timm':
        args = cfg.TRAIN.OPTIMIZER_ARGS

        print(f'=> usage timm optimizer args: {cfg.TRAIN.OPTIMIZER_ARGS}')
        optimizer = create_optimizer(args, model)

        return optimizer

    optimizer = None
    params = set_wd(cfg, model)
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            params,
            # filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            params,
            # filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            weight_decay=cfg.TRAIN.WD,
        )
    elif cfg.TRAIN.OPTIMIZER == 'adamW':
        optimizer = optim.AdamW(
            params,
            lr=cfg.TRAIN.LR,
            weight_decay=cfg.TRAIN.WD,
        )
    elif cfg.TRAIN.OPTIMIZER == 'rmsprop':
        optimizer = optim.RMSprop(
            params,
            # filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            alpha=cfg.TRAIN.RMSPROP_ALPHA,
            centered=cfg.TRAIN.RMSPROP_CENTERED
        )

    return optimizer

