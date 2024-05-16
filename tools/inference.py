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

import skais.utils.colors
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from scipy.spatial import cKDTree
from tqdm import tqdm

from datasets.Illustris import Illustris
from utils.misc import find_galaxy_center, get_mass_histogram

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
    # Iterate through all indices
    for idx in tqdm(idcs):
        sample = ds.dataset.id_list[idx]
        out_fp = ROOT / 'plot' / str(sample["snapshot"]) / str(sample["index"])
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        _, _, pc_arr = ds.dataset[idx]
        # pc_full, _ = ds.dataset._load_sample(sample)
        pc_input_t: torch.Tensor
        pc_gt_t: torch.Tensor
        pc_input_t, pc_gt_t = pc_arr

        # inference
        ret = model(pc_input_t.unsqueeze(0).to(args.device.lower()))
        pc_output_t = ret[-1].squeeze(0).detach().cpu().numpy()

        fig_t, _ = plot_clouds(
            f"DM to Gas (transformed) {sample['model_id']} (epoch {state_dict['epoch']})",
            {
                'DM Input': pc_input_t.numpy(),
                'Gas Ground Truth': pc_gt_t.numpy(),
                '': None,
                'Gas Prediction': pc_output_t,
            }
        )
        fig_t.tight_layout()
        fig_t.show()

        pc_input = rev_transf({'output': pc_input_t.numpy()})['output']
        pc_gt = rev_transf({'output': pc_gt_t.numpy()})['output']
        pc_output: np.ndarray = rev_transf({'output': pc_output_t})['output']
        # pc_input = pc_input_t.numpy()
        # pc_gt = pc_gt_t
        # pc_output = pc_output_t

        fig_dt, _ = plot_clouds(
            f"DM to Gas (retransformed) {sample['model_id']} (epoch {state_dict['epoch']})",
            {
                'DM Input': pc_input,
                'Gas Ground Truth': pc_gt,
                '': None,
                'Gas Prediction': pc_output,
            }
        )
        fig_dt.tight_layout()
        fig_dt.show()

        fig, axs = plt.subplots(2, 2, layout='constrained')
        fig: Figure
        fig.suptitle(f"Masses over distance from galaxy center {sample['model_id']}")
        ax1: Axes = axs[0, 0]
        ax2: Axes = axs[0, 1]
        ax3: Axes = axs[1, 1]
        fig.delaxes(axs[1, 0])

        _ = ax1.hist(np.linalg.norm(pc_input, axis=1), bins=50)
        ax1.set_title("DM input")
        _ = ax2.hist(np.linalg.norm(pc_gt, axis=1), bins=50)
        ax2.set_title("Gas GT")

        _ = ax3.hist(np.linalg.norm(pc_output, axis=1), bins=50)
        ax3.set_title("Gas output")

        for ax in axs.flatten():
            ax.grid()
            ax.set_xlabel('kpc')
            ax.set_ylabel('solMass*')
        # fig.savefig(out_fp.with_suffix('.masses.png'))
        fig.show()

        if args.out_pc_root != '':
            target_path = os.path.join(args.out_pc_root, os.path.splitext(pc_path)[0])
            os.makedirs(target_path, exist_ok=True)

            np.save(os.path.join(target_path, 'fine.npy'), pc_output_t)
        if args.save_vis_img:
            cv2.imwrite(str(out_fp.with_suffix('.dm_input.jpg')), misc.get_ptcloud_img(pc_input))
            cv2.imwrite(str(out_fp.with_suffix('.gas_gt.jpg')), misc.get_ptcloud_img(pc_gt))
            cv2.imwrite(str(out_fp.with_suffix('.gas_output.jpg')), misc.get_ptcloud_img(pc_output))
    return

cmaps = [
    skais.utils.colors.SKAIsCMaps.arcus,
    skais.utils.colors.SKAIsCMaps.phoenix,
    skais.utils.colors.SKAIsCMaps.prism,
    skais.utils.colors.SKAIsCMaps.twofourone,
    skais.utils.colors.SKAIsCMaps.vibes,
    skais.utils.colors.SKAIsCMaps.gaseous,
    skais.utils.colors.SKAIsCMaps.nava,
]
CMAP_DM = skais.utils.colors.SKAIsCMaps.arcus
CMAP_GAS = skais.utils.colors.SKAIsCMaps.gaseous
CMAP_STAR = skais.utils.colors.SKAIsCMaps.phoenix
def plot_clouds(title: str, pcs: dict[str, np.ndarray], log_scale: bool = True) -> tuple[Figure, list[Axes]]:
    rows = math.ceil(len(pcs) / 3)
    columns = min(3, len(pcs))
    fig, axes = plt.subplots(rows, columns)
    fig.set_figheight(10)
    fig.set_figwidth(15)
    fig.suptitle(title)
    cmaps = np.array([CMAP_DM, CMAP_GAS, CMAP_STAR])
    cmaps = np.stack([cmaps for _ in range(columns)], axis=1)
    cmaps = cmaps[:rows, :]
    # if columns > 2:  # nvm, let's not get fancy. It's gotta be easy to compare.
    #     cmaps[:, 2:] = CMAP_STAR
    pc_all = np.concatenate([pc for pc in pcs.values() if pc is not None], axis=0)
    # nan_mask = np.isnan(pc_all).sum(axis=1) > 0
    # pc_all_no_nan = pc_all[~nan_mask]
    xyz_min = pc_all.min(axis=0)
    box_dims = (pc_all.max(axis=0) - pc_all.min(axis=0))
    # box size considering X and Y coords only
    box_size = box_dims[0:2].max().astype(np.float32)
    if isinstance(axes, Axes):
        axes = np.array([axes])
    for ax, pc_data, cmap in zip_longest(axes.flatten(), pcs.items(), cmaps.flatten()):
        if pc_data is None or pc_data[-1] is None:
            fig.delaxes(ax)
            continue
        pc_title, pc = pc_data
        ax.set_title(pc_title)
        # cmap = cmaps.pop(0)
        # print(pc_title, cmap.name)
        plot_cloud(pc, -xyz_min, ax=ax, box_size=box_size, log_scale=log_scale, cmap=cmap)
    return fig, axes.flatten()

def plot_cloud(xyz: np.ndarray, offset: np.ndarray, ax: Axes, box_size: float, mean_mass: float = 89603.26, log_scale: bool = True, cmap: Colormap = None):
    nan_mask = np.isnan(xyz).sum(axis=1) > 0
    if nan_mask.sum() > 0:
        print(f"NaNs: {nan_mask.sum()}")
        xyz = xyz[~nan_mask]
    n_points = xyz.shape[0]
    xyz = xyz + offset
    #box_size = (xyz.max(axis=0) - xyz.min(axis=0)).max().astype(np.float32)
    kdtree = cKDTree(xyz, leafsize=16, boxsize=box_size * 10)
    dist, _ = kdtree.query(xyz, 32, workers=4)
    new_box_size = box_size / 2 * math.sqrt(2)
    xy_offset = new_box_size / 2
    plot = np.zeros((320, 320), dtype=np.double)
    # [f"{n}: {v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}" for n, v in [("mean", xyz.mean(axis=0)), ("min", xyz.min(axis=0)), ("max", xyz.max(axis=0)), ("std", xyz.std(axis=0))]]
    voronoi_RT_2D(
        density=plot,
        pos=np.ascontiguousarray((xyz - offset).astype(np.float32)),
        mass=np.ones((n_points,), dtype=np.float32) * mean_mass,
        radius=np.ascontiguousarray(dist[:, -1].astype(np.float32) * 2),
        x_min=-xy_offset,
        y_min=-xy_offset,
        axis_x=0,
        axis_y=1,
        box_size=new_box_size,
        periodic=False,
        verbose=True,
    )
    if log_scale:
        plot = np.log10(plot)
    ax.imshow(plot, zorder=0, cmap=cmap)
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

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