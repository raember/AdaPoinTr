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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))

from utils.config import cfg_from_yaml_file

ROOT = Path(__file__).parent.parent


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_config', 
        help = 'yaml config file')
    parser.add_argument(
        'model_checkpoint', 
        help = 'pretrained weight')
    parser.add_argument('--pc_root', type=str, default='', help='Pc root')
    parser.add_argument('--pc', type=str, default='', help='Pc file')   
    parser.add_argument(
        '--save_vis_img',
        action='store_true',
        default=False,
        help='whether to save img of complete point cloud') 
    parser.add_argument(
        '--out_pc_root',
        type=str,
        default='',
        help='root of the output pc file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    assert args.save_vis_img or (args.out_pc_root != '')
    assert args.model_config is not None
    assert args.model_checkpoint is not None
    assert (args.pc != '') or (args.pc_root != '')

    return args


def main():
    args = get_args()

    # init config
    config = cfg_from_yaml_file(args.model_config)

    ds_config = config.dataset.train
    ds_config.others.load_to_ram = False

    ds = Illustris(ds_config)
    ds.create_h5_subset()

if __name__ == '__main__':
    main()