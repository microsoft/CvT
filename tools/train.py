from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
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
from config import save_config
from core.loss import build_criterion
from core.function import train_one_epoch, test
from dataset import build_dataloader
from models import build_model
from optim import build_optimizer
from scheduler import build_lr_scheduler
from utils.comm import comm
from utils.utils import create_logger
from utils.utils import init_distributed
from utils.utils import setup_cudnn
from utils.utils import summary_model_on_master
from utils.utils import resume_checkpoint
from utils.utils import save_checkpoint_on_master
from utils.utils import save_model_on_master


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train classification network')

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
    final_output_dir = create_logger(config, args.cfg, 'train')
    tb_log_dir = final_output_dir

    if comm.is_main_process():
        logging.info("=> collecting env info (might take some time)")
        logging.info("\n" + get_pretty_env_info())
        logging.info(pprint.pformat(args))
        logging.info(config)
        logging.info("=> using {} GPUs".format(args.num_gpus))

        output_config_path = os.path.join(final_output_dir, 'config.yaml')
        logging.info("=> saving config into: {}".format(output_config_path))
        save_config(config, output_config_path)

    model = build_model(config)
    model.to(torch.device('cuda'))

    # copy model file
    summary_model_on_master(model, config, final_output_dir, True)

    if config.AMP.ENABLED and config.AMP.MEMORY_FORMAT == 'nhwc':
        logging.info('=> convert memory format to nhwc')
        model.to(memory_format=torch.channels_last)

    writer_dict = {
        'writer': SummaryWriter(logdir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    best_perf = 0.0
    best_model = True
    begin_epoch = config.TRAIN.BEGIN_EPOCH
    optimizer = build_optimizer(config, model)

    best_perf, begin_epoch = resume_checkpoint(
        model, optimizer, config, final_output_dir, True
    )

    train_loader = build_dataloader(config, True, args.distributed)
    valid_loader = build_dataloader(config, False, args.distributed)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
        )

    criterion = build_criterion(config)
    criterion.cuda()
    criterion_eval = build_criterion(config, train=False)
    criterion_eval.cuda()

    lr_scheduler = build_lr_scheduler(config, optimizer, begin_epoch)

    scaler = torch.cuda.amp.GradScaler(enabled=config.AMP.ENABLED)

    logging.info('=> start training')
    for epoch in range(begin_epoch, config.TRAIN.END_EPOCH):
        head = 'Epoch[{}]:'.format(epoch)
        logging.info('=> {} epoch start'.format(head))

        start = time.time()
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        # train for one epoch
        logging.info('=> {} train start'.format(head))
        with torch.autograd.set_detect_anomaly(config.TRAIN.DETECT_ANOMALY):
            train_one_epoch(config, train_loader, model, criterion, optimizer,
                            epoch, final_output_dir, tb_log_dir, writer_dict,
                            scaler=scaler)
        logging.info(
            '=> {} train end, duration: {:.2f}s'
            .format(head, time.time()-start)
        )

        # evaluate on validation set
        logging.info('=> {} validate start'.format(head))
        val_start = time.time()

        if epoch >= config.TRAIN.EVAL_BEGIN_EPOCH:
            perf = test(
                config, valid_loader, model, criterion_eval,
                final_output_dir, tb_log_dir, writer_dict,
                args.distributed
            )

            best_model = (perf > best_perf)
            best_perf = perf if best_model else best_perf

        logging.info(
            '=> {} validate end, duration: {:.2f}s'
            .format(head, time.time()-val_start)
        )

        lr_scheduler.step(epoch=epoch+1)
        if config.TRAIN.LR_SCHEDULER.METHOD == 'timm':
            lr = lr_scheduler.get_epoch_values(epoch+1)[0]
        else:
            lr = lr_scheduler.get_last_lr()[0]
        logging.info(f'=> lr: {lr}')

        save_checkpoint_on_master(
            model=model,
            distributed=args.distributed,
            model_name=config.MODEL.NAME,
            optimizer=optimizer,
            output_dir=final_output_dir,
            in_epoch=True,
            epoch_or_step=epoch,
            best_perf=best_perf,
        )

        if best_model and comm.is_main_process():
            save_model_on_master(
                model, args.distributed, final_output_dir, 'model_best.pth'
            )

        if config.TRAIN.SAVE_ALL_MODELS and comm.is_main_process():
            save_model_on_master(
                model, args.distributed, final_output_dir, f'model_{epoch}.pth'
            )

        logging.info(
            '=> {} epoch end, duration : {:.2f}s'
            .format(head, time.time()-start)
        )

    save_model_on_master(
        model, args.distributed, final_output_dir, 'final_state.pth'
    )

    if config.SWA.ENABLED and comm.is_main_process():
        save_model_on_master(
             args.distributed, final_output_dir, 'swa_state.pth'
        )

    writer_dict['writer'].close()
    logging.info('=> finish training')


if __name__ == '__main__':
    main()
