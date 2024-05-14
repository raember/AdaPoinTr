import pdb
import socket
from copy import deepcopy
from io import BytesIO
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import os
import json

import torchist
import wandb
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch.nn import KLDivLoss, functional, MSELoss
from tqdm import tqdm

from datasets.data_transforms import Compose
from tools import builder
from utils import misc, dist_utils, parser
import time

from utils.config import get_config
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from utils.util import histogram


def run_stats(args, config):
    # build dataset
    load_to_ram = socket.gethostname() != 'apu'
    config.dataset.train.others.load_to_ram = load_to_ram
    config.dataset.val.others.load_to_ram = load_to_ram
    config.dataset.train.others.bs = 1
    config.dataset.val.others.bs = 1
    (train_sampler, train_dataloader), (_, test_dataloader) = builder.dataset_builder(args, config.dataset.train), \
        builder.dataset_builder(args, config.dataset.val)
    i = 0
    for dataloader in [train_dataloader, test_dataloader]:
        kldiv, mse = get_stats(dataloader)
        print(f"KLDIV: mean: {kldiv.mean()}, std: {kldiv.std()}, min: {kldiv.min()}, max: {kldiv.max()}")
        print(f"MSE:   mean: {mse.mean()}, std: {mse.std()}, min: {mse.min()}, max: {mse.max()}")
        plt.plot(range(kldiv.size(0)), torch.sort(kldiv).values.cpu().numpy())
        plt.savefig(f"kldiv_plot_{i}.png")
        plt.plot(range(mse.size(0)), torch.sort(mse).values.cpu().numpy())
        plt.savefig(f"mse_plot_{i}.png")
        i += 1
    print('done')


def get_stats(dataloader) -> Tuple[torch.Tensor, torch.Tensor]:
    kl_divs = []
    mses = []
    for idx, (taxonomy_ids, model_ids, data) in enumerate(tqdm(dataloader)):
        pc_input = data[0].cuda()
        pc_gt = data[1].cuda()
        partial_orig_dist = torch.linalg.norm(pc_input, dim=2)
        gt_orig_dist = torch.linalg.norm(pc_gt, dim=2)
        max_dist = torch.cat([gt_orig_dist, partial_orig_dist], dim=1).max()
        edges = torch.linspace(0, max_dist, 50)
        hist_partial = histogram(partial_orig_dist, edges=edges).to(pc_input.dtype)
        hist_gt = histogram(gt_orig_dist, edges=edges).to(pc_input.dtype)
        kl_div = KLDivLoss(
            reduction='batchmean'
        )(
            input=functional.log_softmax(hist_partial / hist_partial.sum() * 10000, dim=0),
            target=functional.softmax(hist_gt / hist_gt.sum() * 10000, dim=0),
        )  # Faulty because it rescales to max per histogram, but we need softmax distribution -> not a good metric
        # KLDiv DM -> GAS (both gt): ~1.05
        mse = MSELoss(
            reduction='sum'
        )(
            input=hist_partial / hist_partial.sum(),
            target=hist_gt / hist_gt.sum(),
        )
        # MSE DM -> GAS (both gt): ~0.005
        kl_divs.append(kl_div.reshape((1,)))
        mses.append(mse.reshape((1,)))
    return torch.cat(kl_divs), torch.cat(mses)


if __name__ == '__main__':
    args = parser.get_args()
    args.distributed = False
    config = get_config(args)
    run_stats(args, config)
