import os
import pdb

import h5py
from h5py import ExternalLink

import skais
from skais.read import TNGGalaxy
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm

if __name__ == '__main__':
    PC_FOLDER = Path('/cluster/data/ska/illustris/tng50-1.pc')
    PC_FOLDER.mkdir(parents=True, exist_ok=True)

    # with h5py.File(PC_FOLDER / 'galaxy_point_clouds_all_final.hdf5', 'w') as f:
    #     g_data = f.require_group('data')
    #     for snapshot in [50, 78, 99]:
    #         g_data[str(snapshot)] = ExternalLink(
    #             f"galaxy_point_clouds_snapshot_{snapshot}.hdf5",
    #             f"/data/{snapshot}"
    #         )
    # exit()

    GAL_INDICES = range(3333)

    def to_ds(group, title: str, obj):
        d = group.require_dataset(title, obj.shape, dtype='f', compression="gzip")
        d[:] = obj.value.astype(np.float32)
        d.attrs['unit'] = obj.unit.to_string()
        d.attrs['n_points'] = obj.value.shape[0]
        d.attrs['min'] = np.amin(obj.value, 0)
        d.attrs['max'] = np.amax(obj.value, 0)

    with open('../export.log', 'a') as l:
        with h5py.File(PC_FOLDER / 'galaxy_point_clouds_dm2gas_posvel.hdf5', 'a') as f:
            g_data = f.require_group('data')
            for snapshot in [50, 78, 84, 91, 99]:
                g_snapshot = g_data.require_group(str(snapshot))
                for halo_idx in tqdm(GAL_INDICES, dynamic_ncols=True):
                    tqdm.write(f"[{snapshot}] Galaxy #{halo_idx}")
                    l.write(f"[{snapshot}] Galaxy #{halo_idx}\n")
                    if str(halo_idx) in g_snapshot.keys() and 'star' in g_snapshot[str(halo_idx)].keys() and 'max' in g_snapshot[f"{str(halo_idx)}/star"].attrs:
                        tqdm.write(f"[{snapshot}]  -> Skipping: Already exported")
                        l.write(f"[{snapshot}]  -> Skipping: Already exported\n")
                        continue
                    tqdm.write(f"[{snapshot}]  -> Loading galaxy")
                    l.write(f"[{snapshot}]  -> Loading galaxy")
                    dm_gal = TNGGalaxy('tng50-1', snapshot, halo_idx, particle_type='dm')
                    if dm_gal.particle_positions.shape[0] < 10000:
                        tqdm.write(f"[{snapshot}]  -> Skipping: DM too little")
                        l.write(f"[{snapshot}]  -> Skipping: DM too little\n")
                        continue
                    gas_gal = TNGGalaxy('tng50-1', snapshot, halo_idx, particle_type='gas')
                    if gas_gal.particle_positions.shape[0] < 2**14:
                        tqdm.write(f"[{snapshot}]  -> Skipping: GAS too little")
                        l.write(f"[{snapshot}]  -> Skipping: GAS too little\n")
                        continue
                    # star_gal = TNGGalaxy('tng50-1', snapshot, halo_idx, particle_type='star')
                    # if star_gal.particle_positions.shape[0] < 2**14:
                    #     tqdm.write(f"[{snapshot}]  -> Skipping: STAR too little")
                    #     l.write(f"[{snapshot}]  -> Skipping: STAR too little\n")
                    #     continue
                    g_gal = g_snapshot.require_group(str(halo_idx))
                    for par_type, gal in {
                        'dm': dm_gal,
                        'gas': gas_gal,
                        # 'star': star_gal,
                    }.items():
                        g_part = g_gal.require_group(par_type)
                        tqdm.write(f"[{snapshot}]  -> PARTICLE: {par_type}, N={gal.particle_positions.shape[0]}")
                        l.write(f"[{snapshot}]  -> PARTICLE: {par_type}, N={gal.particle_positions.shape[0]}\n")
                        g_part.attrs['center'] = gal.center
                        to_ds(g_part, 'positions', gal.particle_positions - gal.center)
                        # to_ds(g_part, 'masses', gal.masses)
                        to_ds(g_part, 'velocities', gal.velocities)
