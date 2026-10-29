"""Random seeds, distributed setup and logging for semantic training."""
import datetime
import logging
import os
import random

import numpy as np
import torch
from torch.backends import cudnn
from torch import distributed as dist


def fix_seeds(seed: int = 3407) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_cudnn() -> None:
    cudnn.benchmark = True
    cudnn.deterministic = False


def setup_ddp():
    if 'SLURM_PROCID' in os.environ and not 'RANK' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ["SLURM_PROCID"])
        gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
        gpu = rank - gpus_per_node * (rank // gpus_per_node)
        torch.cuda.set_device(gpu)
        dist.init_process_group(backend="nccl", world_size=world_size, rank=rank, timeout=datetime.timedelta(seconds=7200))
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(gpu)
        dist.init_process_group('nccl', init_method="env://",world_size=world_size, rank=rank, timeout=datetime.timedelta(seconds=7200))
        dist.barrier(device_ids=[gpu])
    else:
        gpu = 0
    return gpu


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_logger(log_file=None, file_mode='w'):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: - %(message)s',datefmt='%Y%m%d %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    del logger.handlers[:]

    if log_file:
        file_handler = logging.FileHandler(log_file, mode=file_mode)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    return logger
