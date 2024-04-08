import pdb
from pathlib import Path
from time import sleep
from typing import Tuple, List, Dict

import math

import h5py
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch
import numpy as np
import os, sys

from tqdm import trange, tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import data_transforms
from .io import IO
import json
from .build import DATASETS
from utils.logger import *
from h5py import Group, Dataset


# References:
# - https://github.com/hzxie/GRNet/blob/master/utils/data_loaders.py


def gen_split(h5: h5py.Group) -> list:
    # l = [(snapshot, idx) for snapshot in [50, 78, 99] for idx in range(3333)]
    l = [(snapshot, idx) for snapshot in h5.keys() for idx in h5[snapshot].keys()]
    n_train = int(0.8*len(l))
    n_test = len(l) - n_train
    train, test = data.random_split(l, [n_train, n_test], generator=torch.Generator().manual_seed(42))
    train = [train.dataset[i] for i in train.indices]
    test = [test.dataset[i] for i in test.indices]
    d = []
    for i, taxonomy in enumerate(['dm2gas']):
        d.append({
        "taxonomy_id": str(i),
        "taxonomy_name": taxonomy,
        "test": test,
        "train": train,
        "val": test,
    })
    return d

@DATASETS.register_module()
class Illustris(data.Dataset):
    # def __init__(self, data_root, subset, class_choice = None):
    def __init__(self, config):
        self.data_path = config.DATA_PATH
        self.category_file = config.CATEGORY_FILE_PATH
        self.snapshots = config.SNAPSHOTS
        self.tax_from_n = config.FROM_N
        self.tax_to_n = config.TO_N

        self.n_points = config.N_POINTS
        self.n_renderings = config.N_RENDERINGS
        self.subset = config.subset

        self.load_to_ram = config.load_to_ram if hasattr(config, 'load_to_ram') else False

        # Load the dataset indexing file
        self.h5: Group = IO.get(self.data_path)
        self.dataset_categories = []
        # with open(self.category_file, 'w') as f:
        #     json.dump(gen_split(self.h5), f)
        with open(self.category_file) as f:
            self.dataset_categories = json.load(f)

        # with h5py.File(self.data_path, 'a') as f:
        #     g_data = f['data']
        #     for snapshot in [50, 78, 99]:
        #         g_snapshot = g_data[str(snapshot)]
        #         for halo_idx in trange(3333, dynamic_ncols=True):
        #             g_gal = g_snapshot[str(halo_idx)]
        #             tqdm.write(f"Galaxy #{halo_idx}")
        #             for par_type in ['gas', 'dm', 'star']:
        #                 g_part = g_gal[par_type]
        #                 for d in ['positions', 'masses', 'velocities']:
        #                     ds = g_part[d]
        #                     pc = ds[:]
        #                     ds.attrs['n_points'] = pc.shape[0]
        # print('Done preprocessing')
        # with h5py.File(self.data_path, 'a') as fp:
        #     fp.create_group('data')
        #     for snapshot in [50, 78, 99, 84]:
        #         fp.move(f"{snapshot}", f"data/{snapshot}")
        self.id_list, self.cached_samples = self._get_id_list(self.subset)
        self.transforms = self._get_transforms(self.subset)

    def _get_transforms(self, subset):
        # if subset == 'train':
        #     return data_transforms.Compose([{
        #         'callback': 'ToTensor',
        #         'objects': ['partial_cloud', 'gtcloud']
        #     }])
        # else:
        #     return data_transforms.Compose([{
        #         'callback': 'ToTensor',
        #         'objects': ['partial_cloud', 'gtcloud']
        #     }])
        if subset == 'train':
            return data_transforms.Compose([{
            #     'callback': 'SplatGalaxy',
            #     'parameters': {
            #         'constant': 1000,
            #         'gamma': 2,
            #     },
            #     'objects': ['partial_cloud', 'gtcloud']
            # }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])
        else:
            return data_transforms.Compose([{
            #     'callback': 'SplatGalaxy',
            #     'parameters': {
            #         'constant': 1000,
            #         'gamma': 2,
            #     },
            #     'objects': ['partial_cloud', 'gtcloud']
            # }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])

    def get_taxonomies(self, tax: str) -> Tuple[List[str], List[str]]:
        tax_from, tax_to = tax.split('2', maxsplit=1)
        tax_from = tax_from.split(',')
        tax_to = tax_to.split(',')
        return tax_from, tax_to

    def _get_id_list(self, subset):
        """Prepare id list for the dataset"""
        file_list = []
        discarded = []
        cached_samples = []

        for dc in self.dataset_categories:
            print_log('Collecting IDs of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']), logger='ILLUSTRIS_DATASET')
            samples = dc[subset]
            # min_xyz = np.zeros((3,))
            # max_xyz = np.zeros((3,))
            for (snapshot, idx) in tqdm(samples):
                # tax_from, tax_to = self.get_taxonomies(dc['taxonomy_name'])
                # min_xyz += self.h5[f"{snapshot}/{idx}/dm/positions"].attrs['min']
                # max_xyz += self.h5[f"{snapshot}/{idx}/dm/positions"].attrs['max']
                # n_from = sum([self.h5[f"{snapshot}/{idx}/{tax}/positions"].attrs['n_points'] for tax in tax_from])
                # n_to = sum([self.h5[f"{snapshot}/{idx}/{tax}/positions"].attrs['n_points'] for tax in tax_to])
                # if n_from < self.tax_from_n or n_to < self.tax_to_n:
                #     dm_too_few = n_from < self.tax_from_n
                #     other_too_few = n_to < self.tax_to_n
                #     assert dm_too_few != other_too_few
                #     discarded.append(f"{snapshot}-{idx:04} {tax_to[-1]} does not have enough points: "
                #                      f"{n_from if dm_too_few else n_to} < "
                #                      f"{self.tax_from_n if dm_too_few else (self.tax_to_n - self.tax_from_n)}\n")
                #     continue
                sample = {
                    'taxonomy_id': dc['taxonomy_id'],
                    'taxonomy_name': dc['taxonomy_name'],
                    'model_id': (snapshot, idx),
                    'snapshot': snapshot,
                    'index': idx,
                }
                file_list.append(sample)
                if self.load_to_ram:
                    cached_samples.append(self._load_sample(sample))
            # print_log(f'DM XYZ box length mean: {(max_xyz - min_xyz)/len(samples)}', logger='ILLUSTRIS_DATASET')
        # with open(f'discarded_{subset}.txt', 'w') as fp:
        #     fp.writelines(sorted(discarded))

        print_log('Complete collecting files of the dataset. Total files: %d' % len(file_list), logger='ILLUSTRIS_DATASET')
        return file_list, cached_samples

    def _load_sample(self, sample: dict):
        tax_from, tax_to = self.get_taxonomies(sample['taxonomy_name'])
        try:
            partial_cloud = np.concatenate(
                [self.h5[f"{sample['snapshot']}/{sample['index']}/{tax}/positions"][()] for tax in tax_from])
            # partial_cloud_pos = np.concatenate(
            #     [self.h5[f"{sample['snapshot']}/{sample['index']}/{tax}/positions"][()] for tax in tax_from])
            # partial_cloud_mass = np.concatenate(
            #     [self.h5[f"{sample['snapshot']}/{sample['index']}/{tax}/masses"][()] for tax in tax_from])
            # partial_cloud_mass = np.ones((partial_cloud_pos.shape[0], 1), dtype=partial_cloud_pos.dtype)
            # partial_cloud = np.concatenate([partial_cloud_pos, partial_cloud_mass], axis=1)
            gtcloud = np.concatenate(
                [self.h5[f"{sample['snapshot']}/{sample['index']}/{tax}/positions"][()] for tax in tax_to])
            # gtcloud_pos = np.concatenate(
            #     [self.h5[f"{sample['snapshot']}/{sample['index']}/{tax}/positions"][()] for tax in tax_to])
            # gtcloud_mass = np.concatenate(
            #     [self.h5[f"{sample['snapshot']}/{sample['index']}/{tax}/mass"][()] for tax in tax_to])
            # gtcloud_mass = np.ones((gtcloud_pos.shape[0], 1), dtype=gtcloud_pos.dtype)
            # gtcloud = np.concatenate([gtcloud_pos, gtcloud_mass], axis=1)
        except OSError:
            raise OSError()
            # H5 read error. Possibly because of mounting the cluster
            print_log('Reading for H5 failed. Waiting a couple seconds before retrying.', logger='ILLUSTRIS_DATASET')
            sleep(5)
            self.h5 = IO.get(self.data_path)
            partial_cloud = np.concatenate(
                [self.h5[f"{sample['snapshot']}/{sample['index']}/{tax}/positions"][()] for tax in tax_from])
            gtcloud = np.concatenate(
                [self.h5[f"{sample['snapshot']}/{sample['index']}/{tax}/positions"][()] for tax in tax_to])
        return partial_cloud, gtcloud
        # data = {}
        # partial_cloud = self.random_sample(partial_cloud, self.tax_from_n)
        # gtcloud = self.random_sample(gtcloud, self.tax_to_n - self.tax_from_n)
        # # tax_to = np.concatenate([tax_from, tax_to])
        # data['partial_cloud'] = partial_cloud
        # data['gtcloud'] = gtcloud
        #
        # # assert tax_to.shape[0] == self.tax_to_n, f"{tax_to.shape[0]} != {self.tax_to_n}"
        # # assert tax_from.shape[0] == self.tax_from_n, f"{tax_from.shape[0]} != {self.tax_from_n}"
        # return data

    def create_h5_subset(self):
        h5_path = Path(self.data_path)
        suffix = '_new'
        for dc in self.dataset_categories:
            suffix += f"_dm{self.tax_from_n}_{dc['taxonomy_name']}{self.tax_to_n}"
        h5_path = h5_path.with_name(f"{h5_path.stem}{suffix}{h5_path.suffix}")
        print_log(f'Creating h5 file: {h5_path.name}', logger='ILLUSTRIS_DATASET')
        with h5py.File(str(h5_path), 'a') as fp:
            # fp.create_group('data')
            # for snapshot in [50, 78, 99, 84]:
            #     fp.move(f"{snapshot}", f"data/{snapshot}")
            # return
            for subset in ['train', 'val']:
                discarded = []
                for dc in self.dataset_categories:
                    print_log(f'[{subset}] Collecting IDs of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']), logger='ILLUSTRIS_DATASET')
                    samples = dc[subset]
                    min_xyz = np.zeros((3,))
                    max_xyz = np.zeros((3,))
                    for (snapshot, idx) in tqdm(samples):
                        tax_from, tax_to = self.get_taxonomies(dc['taxonomy_name'])
                        min_xyz += self.h5[f"{snapshot}/{idx}/dm/positions"].attrs['min']
                        max_xyz += self.h5[f"{snapshot}/{idx}/dm/positions"].attrs['max']
                        n_from = sum([self.h5[f"{snapshot}/{idx}/{tax}/positions"].attrs['n_points'] for tax in tax_from])
                        n_to = sum([self.h5[f"{snapshot}/{idx}/{tax}/positions"].attrs['n_points'] for tax in tax_to])
                        if n_from < self.tax_from_n or n_to < self.tax_to_n:
                            dm_too_few = n_from < self.tax_from_n
                            other_too_few = n_to < self.tax_to_n
                            assert dm_too_few != other_too_few
                            discarded.append(f"{snapshot}-{idx:04} {tax_to[-1]} does not have enough points: "
                                             f"{n_from if dm_too_few else n_to} < "
                                             f"{self.tax_from_n if dm_too_few else (self.tax_to_n - self.tax_from_n)}\n")
                            continue
                        for par_type in ['dm', 'gas']: #, 'star']:
                            for dstype in ['positions']: #, 'masses', 'velocities']:
                                fp.create_dataset(f"{snapshot}/{idx}/{par_type}/{dstype}", data=self.h5[f"{snapshot}/{idx}/{par_type}/{dstype}"])
                                for key in self.h5[f"{snapshot}/{idx}/{par_type}/{dstype}"].attrs.keys():
                                    fp[f"{snapshot}/{idx}/{par_type}/{dstype}"].attrs[key] = self.h5[f"{snapshot}/{idx}/{par_type}/{dstype}"].attrs[key]
                            fp[f"{snapshot}/{idx}/{par_type}"].attrs['center'] = self.h5[f"{snapshot}/{idx}/{par_type}"].attrs['center']
                    min_xyz_mean = min_xyz / len(samples)
                    max_xyz_mean = max_xyz / len(samples)
                    print_log(f'DM XYZ box mean length: {max_xyz_mean - min_xyz_mean}', logger='ILLUSTRIS_DATASET')
                    print_log(f'DM XYZ box mean min: {min_xyz_mean}', logger='ILLUSTRIS_DATASET')
                    print_log(f'DM XYZ box mean min: {max_xyz_mean}', logger='ILLUSTRIS_DATASET')
                with open(f'discarded_{subset}.txt', 'w') as fp:
                    fp.writelines(sorted(discarded))

    def __getitem__(self, idx):
        sample = self.id_list[idx]
        if self.load_to_ram:
            partial_cloud, gtcloud = self.cached_samples[idx]
        else:
            partial_cloud, gtcloud = self._load_sample(sample)
        data = {
            'partial_cloud': self.random_sample(partial_cloud, self.tax_from_n).copy() / 1000,
            'gtcloud': self.random_sample(gtcloud, self.tax_to_n).copy() / 1000,
        }
        # from tools.inference import plot_clouds
        # fig, _ = plot_clouds(
        #     'Data pre-transform',
        #     {'Input': data['partial_cloud'], 'GT': data['gtcloud']},
        # )
        # fig.show()
        if self.transforms is not None:
            data = self.transforms(data)
        # data['gtcloud'] = np.concatenate([data['partial_cloud'], data['gtcloud']])  # <- This was a mistake. Don't do this...
        # fig2, _ = plot_clouds(
        #     'Data post-transform',
        #     {
        #         'Input': data['partial_cloud'].numpy(),
        #         'GT': data['gtcloud'].numpy(),
        #     },
        # )
        # fig2.show()
        #
        # rev_transf = data_transforms.Compose([{
        #     'callback': 'BulgeGalaxy',
        #     'parameters': {
        #         'gamma': self.transforms.transformers[0]['callback'].gamma,
        #         'constant': self.transforms.transformers[0]['callback'].c,
        #     },
        #     'objects': ['output']
        # }])
        # fig3, _ = plot_clouds(
        #     'Data re-transform',
        #     {
        #         'Input': rev_transf({'output': data['partial_cloud']})['output'].numpy(),
        #         'GT': rev_transf({'output': data['gtcloud']})['output'].numpy(),
        #     },
        # )
        # fig3.show()
        return sample['taxonomy_id'], sample['model_id'], (data['partial_cloud'], data['gtcloud'])

    def random_sample(self, points: np.ndarray, n: int) -> np.ndarray:
        return points[np.random.permutation(points.shape[0])[:n], ...]

    def get(self, idx: int) -> Dict[str, np.ndarray]:
        sample = self.id_list[idx]
        if self.load_to_ram:
            partial_cloud, gtcloud = self.cached_samples[idx]
        else:
            partial_cloud, gtcloud = self._load_sample(sample)
        data = {
            'partial_cloud': partial_cloud / 1000,
            'gtcloud': gtcloud / 1000,
        }
        return data

    def __len__(self):
        return len(self.id_list)
