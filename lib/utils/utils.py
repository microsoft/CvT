from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import timedelta
from pathlib import Path

import os
import logging
import shutil
import time

import tensorwatch as tw
import torch
import torch.backends.cudnn as cudnn

from utils.comm import comm
from ptflops import get_model_complexity_info


def setup_logger(final_output_dir, rank, phase):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_rank{}.txt'.format(phase, time_str, rank)
    final_log_file = os.path.join(final_output_dir, log_file)
    head = '%(asctime)-15s:[P:%(process)d]:' + comm.head + ' %(message)s'
    logging.basicConfig(
        filename=str(final_log_file), format=head
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setFormatter(
        logging.Formatter(head)
    )
    logging.getLogger('').addHandler(console)


def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    dataset = cfg.DATASET.DATASET
    cfg_name = cfg.NAME

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {} ...'.format(root_output_dir))
    root_output_dir.mkdir(parents=True, exist_ok=True)
    print('=> creating {} ...'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    print('=> setup logger ...')
    setup_logger(final_output_dir, cfg.RANK, phase)

    return str(final_output_dir)


def init_distributed(args):
    args.num_gpus = int(os.environ["WORLD_SIZE"]) \
        if "WORLD_SIZE" in os.environ else 1
    args.distributed = args.num_gpus > 1

    if args.distributed:
        print("=> init process group start")
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
            timeout=timedelta(minutes=180))
        comm.local_rank = args.local_rank
        print("=> init process group end")


def setup_cudnn(config):
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def summary_model_on_master(model, config, output_dir, copy):
    if comm.is_main_process():
        this_dir = os.path.dirname(__file__)
        shutil.copy2(
            os.path.join(this_dir, '../models', config.MODEL.NAME + '.py'),
            output_dir
        )
        logging.info('=> {}'.format(model))
        try:
            num_params = count_parameters(model)
            logging.info("Trainable Model Total Parameter: \t%2.1fM" % num_params)
        except Exception:
            logging.error('=> error when counting parameters')

        if config.MODEL_SUMMARY:
            try:
                logging.info('== model_stats by tensorwatch ==')
                df = tw.model_stats(
                    model,
                    (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
                )
                df.to_html(os.path.join(output_dir, 'model_summary.html'))
                df.to_csv(os.path.join(output_dir, 'model_summary.csv'))
                msg = '*'*20 + ' Model summary ' + '*'*20
                logging.info(
                    '\n{msg}\n{summary}\n{msg}'.format(
                        msg=msg, summary=df.iloc[-1]
                    )
                )
                logging.info('== model_stats by tensorwatch ==')
            except Exception:
                logging.error('=> error when run model_stats')

            try:
                logging.info('== get_model_complexity_info by ptflops ==')
                macs, params = get_model_complexity_info(
                    model,
                    (3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0]),
                    as_strings=True, print_per_layer_stat=True, verbose=True
                )
                logging.info(f'=> FLOPs: {macs:<8}, params: {params:<8}')
                logging.info('== get_model_complexity_info by ptflops ==')
            except Exception:
                logging.error('=> error when run get_model_complexity_info')


def resume_checkpoint(model,
                      optimizer,
                      config,
                      output_dir,
                      in_epoch):
    best_perf = 0.0
    begin_epoch_or_step = 0

    checkpoint = os.path.join(output_dir, 'checkpoint.pth')\
        if not config.TRAIN.CHECKPOINT else config.TRAIN.CHECKPOINT
    if config.TRAIN.AUTO_RESUME and os.path.exists(checkpoint):
        logging.info(
            "=> loading checkpoint '{}'".format(checkpoint)
        )
        checkpoint_dict = torch.load(checkpoint, map_location='cpu')
        best_perf = checkpoint_dict['perf']
        begin_epoch_or_step = checkpoint_dict['epoch' if in_epoch else 'step']
        state_dict = checkpoint_dict['state_dict']
        model.load_state_dict(state_dict)

        optimizer.load_state_dict(checkpoint_dict['optimizer'])
        logging.info(
            "=> {}: loaded checkpoint '{}' ({}: {})"
            .format(comm.head,
                    checkpoint,
                    'epoch' if in_epoch else 'step',
                    begin_epoch_or_step)
        )

    return best_perf, begin_epoch_or_step


def save_checkpoint_on_master(model,
                              *,
                              distributed,
                              model_name,
                              optimizer,
                              output_dir,
                              in_epoch,
                              epoch_or_step,
                              best_perf):
    if not comm.is_main_process():
        return

    states = model.module.state_dict() \
        if distributed else model.state_dict()

    logging.info('=> saving checkpoint to {}'.format(output_dir))
    save_dict = {
        'epoch' if in_epoch else 'step': epoch_or_step + 1,
        'model': model_name,
        'state_dict': states,
        'perf': best_perf,
        'optimizer': optimizer.state_dict(),
    }

    try:
        torch.save(save_dict, os.path.join(output_dir, 'checkpoint.pth'))
    except Exception:
        logging.error('=> error when saving checkpoint!')


def save_model_on_master(model, distributed, out_dir, fname):
    if not comm.is_main_process():
        return

    try:
        fname_full = os.path.join(out_dir, fname)
        logging.info(f'=> save model to {fname_full}')
        torch.save(
            model.module.state_dict() if distributed else model.state_dict(),
            fname_full
        )
    except Exception:
        logging.error('=> error when saving checkpoint!')


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    from collections import OrderedDict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict
