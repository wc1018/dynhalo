from typing import Tuple, List, Union

import h5py as h5
import numpy as np
import pandas as pd

from dynhalo.utils import G_gravity
from dynhalo.finder.subbox import load_particles, load_seeds
from dynhalo.finder.coordinates import relative_coordinates


def find_r200_m200(
    pos: np.ndarray,
    part_mass: float,
    rhom: float,
) -> Tuple[float]:
    """Find R200 and M200 around a given seed using all particles.

    Parameters
    ----------
    pos : np.ndarray
        Relative coordinates
    part_mass : float
        Mass per particle
    rhom : float
        Matter density of the universe

    Returns
    -------
    Tuple[float]
        R200 and M200
    """
    dists = np.sqrt(np.sum(np.square(pos), axis=1))
    dists.sort()
    for i in range(len(dists)):
        density = ((i + 1) * part_mass) / (4 / 3 * np.pi * (dists[i] ** 3))
        if density <= rhom * 200:
            return (dists[i], (i + 1) * part_mass)
    # Seed has no free particles around it
    return (None, None)


def classify(
    rel_pos: np.ndarray,
    rel_vel: np.ndarray,
    r200: float,
    m200: float,
    class_pars: Union[List, Tuple, np.ndarray],
) -> np.ndarray:
    """Classifies particles as orbiting.

    Parameters
    ----------
    rel_pos : np.ndarray
        Relative position of particles around seed
    rel_vel : np.ndarray
        Relative velocity of particles around seed
    r200 : float
        Seed R200
    m200 : float
        Seed M200
    class_pars : Union[List, Tuple, np.ndarray]
        Classification parameters [m_pos, b_pos, m_neg, b_neg]

    Returns
    -------
    np.ndarray
        A boolean array where True == orbiting
    """
    m_pos, b_pos, m_neg, b_neg = class_pars
    # Compute V200
    v200 = G_gravity * m200 / r200

    # Compute the radius to seed_i in r200 units, and ln(v^2) in v200 units
    part_ln_vel = np.log(np.sum(np.square(rel_vel), axis=1) / v200)
    part_radius = np.sqrt(np.sum(np.square(rel_pos), axis=1)) / r200

    # Create a mask for particles with positive radial velocity
    mask_vr_positive = np.sum(rel_vel * rel_pos, axis=1) > 0

    # Orbiting classification for vr > 0
    mask_cut_pos = part_ln_vel < (m_pos * part_radius + b_pos)

    # Orbiting classification for vr < 0
    mask_cut_neg = part_ln_vel < (m_neg * part_radius + b_neg)

    # Particle is infalling if it is below both lines and 2*R00
    mask_orb = \
        (mask_cut_pos & mask_vr_positive) ^ \
        (mask_cut_neg & ~mask_vr_positive)

    return mask_orb


def classify_seeds_in_sub_box(
    sub_box_id: int,
    min_num_part: int,
    part_mass: float,
    rhom: float,
    boxsize: float,
    subsize: float,
    path: str,
    padding: float = 5.0,
) -> None:
    """_summary_

    Parameters
    ----------
    sub_box_id : int
        Sub-box ID
    min_num_part : int
        Minimum number of particles needed to be considered a halo
    part_mass : float
        Mass per particle
    rhom : float
        Matter density of the universe
    boxsize : float
        Size of simulation box
    subsize : float
        Size of sub-box
    path : str
        Location from where to load the file
    padding : float
        Only particles up to this distance from the sub-box edge are considered 
        for classification. Defaults to 5

    Returns
    -------
    None
    """
    # Load seeds
    pos_seed, vel_seed, hid_seed, row_seed = load_seeds(sub_box_id, boxsize,
                                                        subsize, path, padding)
    n_seeds = len(hid_seed)
    # Load adjacent seeds
    pos_seed_adj, vel_seed_adj, hid_seed_adj, row_seed_adj = load_seeds(
        sub_box_id, boxsize, subsize, path, padding, adjacent=True)
    # Concatenate seeds
    hid_seed_adj = np.hstack([hid_seed, hid_seed_adj])
    row_seed_adj = np.hstack([row_seed, row_seed_adj])
    pos_seed_adj = np.vstack([pos_seed, pos_seed_adj])
    vel_seed_adj = np.hstack([vel_seed, vel_seed_adj])

    # Load particles
    pos_part, vel_part, pid_part, row_part = load_particles(sub_box_id, boxsize,
                                                            subsize, path,
                                                            padding)

    # Create empty catalog of found halos and a dictionary with the PIDs
    halo_members = {}
    halo_subs = {}
    col_names = ("OHID", "pos", "vel", "R200m", "M200m", "Morb")
    dtypes = (np.uint32, np.float32, np.float32,
              np.float32, np.float32, np.float32)
    haloes = pd.DataFrame(columns=col_names)

    # Load calibration parameters
    with h5.File(path + 'calibration_pars.hdf5', 'r') as hdf:
        m_pos, b_pos = hdf['pos'][()]
        m_neg, b_neg = hdf['neg'][()]
    pars = (m_pos, b_pos, m_neg, b_neg)

    for i in range(n_seeds):
        # Classify particles ===================================================
        rel_pos = relative_coordinates(pos_seed[i], pos_part)
        rel_vel = vel_part - vel_seed[i]
        r200, m200 = find_r200_m200(rel_pos, part_mass, rhom)
        v200sq = G_gravity * m200 / r200
        # Classify
        mask_orb = classify(rel_pos, rel_vel, r200, m200, pars)
        # Compute phase space distance from particle to halo
        dphsq = np.sum(np.square(rel_pos), axis=1) / r200**2 + \
            np.sum(np.square(rel_vel), axis=1) / v200sq

        # Ignore seed if it does not have the minimum mass to be considered a
        # halo. Early exit to avoid further computation for a non-halo seed
        if mask_orb.sum() < min_num_part:
            continue

        # Select orbiting particles' PID
        orb_pid = pid_part[mask_orb]
        orb_arg = row_part[mask_orb]
        orb_dph = dphsq[mask_orb]

        # Classify seeds =======================================================
        rel_pos = relative_coordinates(pos_seed[i], pos_seed_adj)
        rel_vel = vel_seed_adj - vel_seed[i]
        # Classify
        mask_orb_seed = classify(rel_pos, rel_vel, r200, m200, pars)
        # Compute phase space distance from particle to halo
        dphsq = np.sum(np.square(rel_pos), axis=1) / r200**2 + \
            np.sum(np.square(rel_vel), axis=1) / v200sq
        # Select orbiting particles' PID
        orb_pid_seed = hid_seed_adj[mask_orb_seed]
        orb_arg_seed = row_seed_adj[mask_orb_seed]
        orb_dph_seed = dphsq[mask_orb_seed]

        # Append halo to halo catalogue if there are at least min_dm_part
        # orbiting particles
        haloes.loc[len(haloes.index)] = [
            hid_seed[i],
            pos_seed[i],
            vel_seed[i],
            r200,
            m200,
            part_mass * mask_orb.sum(),
        ]
        halo_members[hid_seed[i]] = np.array([orb_pid, orb_arg, orb_dph]).T

        if mask_orb_seed.sum() > 0:
            halo_subs[hid_seed[i]] = np.array(
                [orb_pid_seed, orb_arg_seed, orb_dph_seed]).T

    # Save catalogue
    with h5.File(path + '', 'w') as hdf:
        # Save halo catalogue
        for i, key in enumerate(haloes.columns):
            data = haloes[key].values
            hdf.create_dataset(f'halo/{key}', data=data, dtype=dtypes[i])
        # Save halo members (particles)
        # Dataset names must be strings (thus are not properly sorted)
        for item, data in halo_members.items():
            hdf.create_dataset(f'members/part/{str(item)}', data=data)
        # Save halo members (seed)
        for item, data in halo_subs.items():
            hdf.create_dataset(f'members/halo/{str(item)}', data=data)

    return None


if __name__ == "__main__":
    pass
