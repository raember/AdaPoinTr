##############################################################
# % Author: Castle
# % Date:14/01/2023
###############################################################
# tools/inference.py --save_vis_img --pc 0 cfgs/Illustris/AdaPoinTr.yaml ckpt-best.pth
import argparse
import os
import sys
from pathlib import Path

from datasets.Illustris import Illustris
from tools import builder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))

from utils.config import cfg_from_yaml_file

ROOT = Path(__file__).parent.parent


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config',
        help = 'yaml config file')
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # init config
    config = cfg_from_yaml_file(args.config)
    ds_config = config.dataset.test
    ds_config.others.load_to_ram = False
    args.distributed = False
    args.num_workers = 4
    _, ds = builder.dataset_builder(args, ds_config)

    il: Illustris = ds.dataset
    il.create_h5_subset()

if __name__ == '__main__':
    main()