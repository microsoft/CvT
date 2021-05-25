from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import pickle as pkl
import pprint
import time

import torch
import torch.nn.parallel
import torch.optim
from torch.utils.collect_env import get_pretty_env_info
from tensorboardX import SummaryWriter

import _init_paths
from config import config
from config import update_config
from core.function import test
from core.loss import build_criterion
from dataset import build_dataloader
from dataset import RealLabelsImagenet
from models import build_model
from utils.comm import comm
from utils.utils import create_logger
from utils.utils import init_distributed
from utils.utils import setup_cudnn
from utils.utils import summary_model_on_master
from utils.utils import strip_prefix_if_present


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test classification network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    # distributed training
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--port", type=int, default=9000)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    init_distributed(args)
    setup_cudnn(config)

    update_config(config, args)
    final_output_dir = create_logger(config, args.cfg, 'test')
    tb_log_dir = final_output_dir

    if comm.is_main_process():
        logging.info("=> collecting env info (might take some time)")
        logging.info("\n" + get_pretty_env_info())
        logging.info(pprint.pformat(args))
        logging.info(config)
        logging.info("=> using {} GPUs".format(args.num_gpus))

        output_config_path = os.path.join(final_output_dir, 'config.yaml')
        logging.info("=> saving config into: {}".format(output_config_path))

    model = build_model(config)
    model.to(torch.device('cuda'))

    model_file = config.TEST.MODEL_FILE if config.TEST.MODEL_FILE \
        else os.path.join(final_output_dir, 'model_best.pth')
    logging.info('=> load model file: {}'.format(model_file))
    ext = model_file.split('.')[-1]
    if ext == 'pth':
        state_dict = torch.load(model_file, map_location="cpu")
    else:
        raise ValueError("Unknown model file")

    model.load_state_dict(state_dict, strict=False)
    model.to(torch.device('cuda'))

    writer_dict = {
        'writer': SummaryWriter(logdir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    summary_model_on_master(model, config, final_output_dir, False)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )

    # define loss function (criterion) and optimizer
    criterion = build_criterion(config, train=False)
    criterion.cuda()

    valid_loader = build_dataloader(config, False, args.distributed)
    real_labels = None
    if (
        config.DATASET.DATASET == 'imagenet'
        and config.DATASET.DATA_FORMAT == 'tsv'
        and config.TEST.REAL_LABELS
    ):
        filenames = valid_loader.dataset.get_filenames()
        real_json = os.path.join(config.DATASET.ROOT, 'real.json')
        logging.info('=> loading real labels...')
        real_labels = RealLabelsImagenet(filenames, real_json)

    valid_labels = None
    if config.TEST.VALID_LABELS:
        with open(config.TEST.VALID_LABELS, 'r') as f:
            valid_labels = {
                int(line.rstrip()) for line in f
            }
            valid_labels = [
                i in valid_labels for i in range(config.MODEL.NUM_CLASSES)
            ]

    logging.info('=> start testing')
    start = time.time()
    test(config, valid_loader, model, criterion,
         final_output_dir, tb_log_dir, writer_dict,
         args.distributed, real_labels=real_labels,
         valid_labels=valid_labels)
    logging.info('=> test duration time: {:.2f}s'.format(time.time()-start))

    writer_dict['writer'].close()
    logging.info('=> finish testing')


if __name__ == '__main__':
    main()
