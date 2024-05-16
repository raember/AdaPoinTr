##############################################################
# % Author: Castle
# % Date:14/01/2023
###############################################################
# tools/inference.py --save_vis_img --pc 0 cfgs/Illustris/AdaPoinTr.yaml ckpt-best.pth
import argparse
import os
from itertools import zip_longest
from pathlib import Path

import math
import numpy as np
import cv2
import sys

import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.spatial import cKDTree
from torch.nn import KLDivLoss, MSELoss, functional
from tqdm import tqdm

from datasets.Illustris import Illustris
from tools.inference import plot_clouds
from utils.misc import find_galaxy_center, get_mass_histogram
from utils.util import histogram

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))

from tools import builder
from utils.config import cfg_from_yaml_file
from utils import misc
from datasets.io import IO
from datasets.data_transforms import Compose, SplatGalaxy
from skais.raytrace import voronoi_RT_2D

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

    assert args.model_config is not None
    assert args.model_checkpoint is not None
    assert (args.pc != '') or (args.pc_root != '')

    return args

def inference_single(model, pc_path, args, config, root=None):
    if root is not None:
        pc_file = os.path.join(root, pc_path)
    else:
        pc_file = pc_path
    state_dict = torch.load(args.model_checkpoint, map_location='cpu')

    ds_config = config.dataset.test
    ds_config.others.load_to_ram = False
    args.distributed = False
    args.num_workers = 4
    _, ds = builder.dataset_builder(args, ds_config)

    does_splat = type(ds.dataset.transforms.transformers[0]['callback']).__name__ == 'SplatGalaxy'
    if does_splat:
        rev_transf = Compose([{
            'callback': 'BulgeGalaxy',
            'parameters': {
                'gamma': ds.dataset.transforms.transformers[0]['callback'].gamma,
                'constant': ds.dataset.transforms.transformers[0]['callback'].c,
            },
            'objects': ['output']
        }])
    else:
        def pass_along(pc: np.ndarray) -> np.ndarray:
            return pc
        rev_transf = pass_along

    if pc_file == '-':
        idcs = range(len(ds.dataset.id_list))
    else:
        idcs = range(len(ds.dataset.id_list))
    want = [
        (55, 800),
        (78, 675),
        (78, 1939),
    ]
    # if len(want) > 0:
    #     idcs = []
    #     for ss, hi in want:
    #         tpl = (str(ss), str(hi))
    #         key = ', '.join(tpl)
    #         for i, sample in enumerate(ds.dataset.id_list):
    #             sample_id = ', '.join(sample['model_id'])
    #             print(i, sample_id)
    #             if sample_id == key:
    #                 idcs.append(i)
    #                 break

    idcs = [
        0,  # triple merger
        # 1,  # beautiful simple galaxy
        2,  # well predicted simple galaxy
        # 5,  # simple galaxy, bad details
        7,  # pointy galaxy, bad details and overall
    ]

    # Iterate through all indices
    for idx in tqdm(idcs):
        sample = ds.dataset.id_list[idx]
        out_fp = ROOT / 'plot' / str(sample["snapshot"]) / str(sample["index"])
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        data = ds.dataset.get(idx)
        idcs_in = np.random.permutation(data['partial_cloud'].shape[0])
        idcs_gt = np.random.permutation(data['gtcloud'].shape[0])
        pc_in = []
        pc_idcs_in = []
        while len(idcs_in) >= ds.dataset.tax_from_n:
            idcs = idcs_in[:ds.dataset.tax_from_n]
            idcs_in = idcs_in[ds.dataset.tax_from_n:]
            sub_data = ds.dataset.transforms({
                'partial_cloud': data['partial_cloud'][idcs],
                # 'gtcloud': data['gtcloud'][idcs],
            })
            sub_pc = sub_data['partial_cloud']
            # pc_in.append(rev_transf({'output': sub_pc.numpy()})['output'])
            pc_in.append(sub_pc)
            pc_idcs_in.append(idcs.copy())
        pc_gt = []
        pc_idcs_gt = []
        while len(idcs_gt) >= ds.dataset.tax_to_n:
            idcs = idcs_gt[:ds.dataset.tax_to_n]
            idcs_gt = idcs_gt[ds.dataset.tax_to_n:]
            sub_data = ds.dataset.transforms({
                # 'partial_cloud': data['partial_cloud'][idcs],
                'gtcloud': data['gtcloud'][idcs],
            })
            sub_pc = sub_data['gtcloud']
            # pc_gt.append(rev_transf({'output': sub_pc.numpy()})['output'])
            pc_gt.append(sub_pc)
            pc_idcs_gt.append(idcs.copy())
        pc_len = min(len(pc_in), len(pc_gt))
        pc_outputs = []
        first = True
        for i in range(pc_len):
            sub_data = ds.dataset.transforms({
                'partial_cloud': data['partial_cloud'][pc_idcs_in[i]],
                'gtcloud': data['gtcloud'][pc_idcs_gt[i]],
            })
            # inference
            partial = sub_data['partial_cloud'].unsqueeze(0).to(args.device.lower())
            gt = sub_data['gtcloud'].unsqueeze(0).to(args.device.lower())
            ret = model(partial)
            dense_points = ret[-1]
            pc_output_t = ret[-1].squeeze(0).detach().cpu().numpy()
            pc_outputs.append(pc_output_t)
            if first:
                # Calculate histogram and metrics
                pc_input = partial
                pc_output = dense_points
                partial_orig_dist = torch.linalg.norm(pc_input, dim=2)
                dense_orig_dist = torch.linalg.norm(pc_output, dim=2)
                gt_orig_dist = torch.linalg.norm(gt, dim=2)
                max_dist = torch.cat([gt_orig_dist, partial_orig_dist], dim=1).max()
                edges = torch.linspace(0, max_dist, 50)
                hist_partial = histogram(partial_orig_dist, edges=edges).to(dense_points.dtype)
                hist_gt = histogram(gt_orig_dist, edges=edges).to(dense_points.dtype)
                hist_dense = histogram(dense_orig_dist, edges=edges).to(dense_points.dtype)
                kl_div = KLDivLoss(
                    reduction='batchmean'
                )(
                    input=functional.log_softmax(hist_dense / hist_dense.sum() * 10000, dim=0),
                    target=functional.softmax(hist_gt / hist_gt.sum() * 10000, dim=0),
                )
                # KLDiv DM -> GAS (both gt): ~1.05
                mse = MSELoss(
                    reduction='sum'
                )(
                    input=hist_dense / hist_dense.sum(),
                    target=hist_gt / hist_gt.sum(),
                )
                tqdm.write(f'KLDiv: {kl_div}, MSE: {mse}')
                fig2, axs = plt.subplots(2, 1, layout='constrained')
                fig2: Figure
                fig2.suptitle(
                    f"Particle counts over distance from galaxy center (epoch={state_dict['epoch']}, [{', '.join(sample['model_id'])}])\nKLDiv: {kl_div:.3e}, MSE: {mse:.3e}")
                ax1: Axes = axs[0]
                ax2: Axes = axs[1]

                ax1.set_title('DM input')
                ax1.stairs(hist_partial.detach().cpu(), edges, fill=True)  # DM input
                ax1.grid()

                ax2.set_title('Gas gt & output')
                ax2.stairs(hist_gt.detach().cpu(), edges, fill=True, zorder=-1)  # Gas gt
                ax2.stairs(hist_dense.detach().cpu(), edges, hatch='//', zorder=0)  # Gas output
                ax2.grid()

                fig2.show()
            first = False

        fig_dt, _ = plot_clouds(
            f"DM to Gas (accumulated PC) {sample['model_id']} (epoch {state_dict['epoch']})",
            {
                f'DM Full [{len(pc_in) * ds.dataset.tax_from_n}]': rev_transf({'output': torch.concatenate(pc_in).numpy()})['output'],
                f'DM Input [{pc_len * ds.dataset.tax_from_n}]': rev_transf({'output': torch.concatenate(pc_in[:pc_len]).numpy()})['output'],
                '': None,
                f'Gas GT Full [{len(pc_gt) * ds.dataset.tax_to_n}]': rev_transf({'output': torch.concatenate(pc_gt).numpy()})['output'],
                f'Gas GT Output [{pc_len * ds.dataset.tax_to_n}]': rev_transf({'output': torch.concatenate(pc_gt[:pc_len]).numpy()})['output'],
                f'Gas Prediction [{pc_len * ds.dataset.tax_to_n}]': rev_transf({'output': np.concatenate(pc_outputs)})['output'],
            }
        )
        fig_dt.tight_layout()
        fig_dt.show()
        print('=' * 20)
    return


def main():
    args = get_args()

    # init config
    config = cfg_from_yaml_file(args.model_config)
    # build model
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.model_checkpoint)
    base_model.to(args.device.lower())
    base_model.eval()

    if args.pc_root != '':
        pc_file_list = os.listdir(args.pc_root)
        for pc_file in pc_file_list:
            inference_single(base_model, pc_file, args, config, root=args.pc_root)
    else:
        inference_single(base_model, args.pc, args, config)

if __name__ == '__main__':
    main()