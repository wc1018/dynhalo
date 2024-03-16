from typing import Tuple

import h5py as h5
import numpy as np
from tqdm import tqdm

from dynhalo.finder.coordinates import (get_vr_vt_from_coordinates,
                                        relative_coordinates)
from dynhalo.finder.halo import find_r200_m200
from dynhalo.finder.subbox import get_sub_box_id, load_particles
from dynhalo.utils import G_gravity, timer


@timer
def _select_particles_around_haloes(
    n_seeds: int,
    r_max: float,
    boxsize: float,
    subsize: float,
    file_seeds: str,
    path: str,
    part_mass: float,
    rhom: float,
) -> Tuple[np.ndarray]:
    """Locates for the largest `v_max` seeds and searches for all the particles
    around them up to a distance `r_max`.

    Parameters
    ----------
    n_seeds : int
        Number of seeds to process
    r_max : float
        Maximum distance to consider
    boxsize : float
        Size of simulation box
    subsize : float
        Size of sub-box
    file_seeds : str
        File containing the seeds, including path.
    path : str
        Path to the sub-boxes
    part_mass : float
        Mass per particle
    rhom : float
        Mass density

    Returns
    -------
    Tuple[np.ndarray]
        Radial distance, radial velocity, and log of the square of the velocity
    """
    # Load seed data
    with h5.File(file_seeds, 'r') as hdf:
        vmax = hdf['vmax'][()]
        order = np.argsort(vmax)[::-1]

        hid = hdf['Orig_halo_ID'][()]
        pos_seed = np.vstack(
            [
                hdf['x'][()],
                hdf['y'][()],
                hdf['z'][()],
            ]
        ).T
        vel_seed = np.vstack(
            [
                hdf['vx'][()],
                hdf['vy'][()],
                hdf['vz'][()],
            ]
        ).T
    vmax = vmax[order][:n_seeds]
    hid = hid[order][:n_seeds]
    pos_seed = pos_seed[order][:n_seeds]
    vel_seed = vel_seed[order][:n_seeds]

    # Locate sub-box IDs for all seeds.
    seed_sub_box_id = get_sub_box_id(pos_seed, boxsize, subsize)
    # Sort by sub-box ID
    order = np.argsort(seed_sub_box_id)
    seed_sub_box_id = seed_sub_box_id[order]
    vmax = vmax[order]
    hid = hid[order]
    pos_seed = pos_seed[order]
    vel_seed = vel_seed[order]

    # Get unique sub-box ids
    unique_sub_box_ids = np.unique(seed_sub_box_id)

    # Create empty lists (containers) to save the data from file for each ID
    r, vr, lnv2 = ([] for _ in range(3))
    # Iterate over sub-box IDs
    # NOTE: Could parallelise this but it is not super slow
    for sub_box in tqdm(unique_sub_box_ids, desc='Processing sub-box',
                        colour='blue', ncols=100):
        pos, vel, _, _ = load_particles(sub_box, boxsize, subsize, path)

        # Iterate over seeds in current sub-box ID
        mask_seeds_in_sub_box = seed_sub_box_id == sub_box
        for i in range(mask_seeds_in_sub_box.sum()):
            # Compute the relative positions of all particles in the box
            rel_pos = relative_coordinates(pos_seed[mask_seeds_in_sub_box][i], pos,
                                           boxsize)
            # Only work with those close to the seed
            mask_x = np.abs(rel_pos[:, 0]) <= r_max
            mask_y = np.abs(rel_pos[:, 1]) <= r_max
            mask_z = np.abs(rel_pos[:, 2]) <= r_max
            mask_close = mask_x * mask_y * mask_z

            # pos_i = pos[mask_close]
            rel_pos = rel_pos[mask_close]
            rel_vel = vel[mask_close] - vel_seed[mask_seeds_in_sub_box][i]
            # Compute R200 and M200
            r200, m200 = find_r200_m200(rel_pos, part_mass, rhom)
            # Compute V200
            v200sq = G_gravity * m200 / r200
            # Compute radial and tangential velocity
            vrp, _, v2p = get_vr_vt_from_coordinates(rel_pos, rel_vel)
            # Resale by v200
            vrp /= np.sqrt(v200sq)
            v2p /= v200sq
            # Compute the radial separation rescaled by R200
            rp = np.sqrt(np.sum(np.square(rel_pos), axis=1)) / r200

            # Append to containers
            r.append(rp)
            vr.append(vrp)
            lnv2.append(np.log(v2p))

    # Concatenate into a single array
    r = np.concatenate(r)
    vr = np.concatenate(vr)
    lnv2 = np.concatenate(lnv2)

    return r, vr, lnv2


def get_calibration_data(
    n_seeds: int,
    r_max: float,
    boxsize: float,
    subsize: float,
    file_seeds: str,
    path: str,
    part_mass: float,
    rhom: float,
) -> Tuple[np.ndarray]:
    """_summary_

    Parameters
    ----------
    n_seeds : int
        Number of seeds to process
    r_max : float
        Maximum distance to consider
    boxsize : float
        Size of simulation box
    subsize : float
        Size of sub-box
    file_seeds : str
        File containing the seeds, including path.
    path : str
        Path to the sub-boxes
    part_mass : float
        Mass per particle
    rhom : float
        Mass density

    Returns
    -------
    Tuple[np.ndarray]
        Radial distance, radial velocity, and log of the square of the velocity
    """
    file_name = path + 'calibration_data.hdf5'
    try:
        with h5.File(file_name, 'r') as hdf:
            r = hdf['r'][()]
            vr = hdf['vr'][()]
            lnv2 = hdf['lnv2'][()]
        return r, vr, lnv2
    except:
        r, vr, lnv2 = _select_particles_around_haloes(
            n_seeds,
            r_max,
            boxsize,
            subsize,
            file_seeds,
            path,
            part_mass,
            rhom,
        )

        with h5.File(file_name, 'w') as hdf:
            hdf.create_dataset('r', data=r)
            hdf.create_dataset('vr', data=vr)
            hdf.create_dataset('lnv2', data=lnv2)

        return r, vr, lnv2
    

if __name__ == "__main__":
    pass
