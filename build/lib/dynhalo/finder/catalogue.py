import os
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from typing import List, Tuple, Union
from warnings import filterwarnings

import h5py as h5
import numba
import numpy as np
import pandas as pd
from tqdm import tqdm

from dynhalo.finder.coordinates import relative_coordinates
from dynhalo.finder.subbox import load_particles, load_seeds
from dynhalo.utils import G_gravity, timer

filterwarnings('ignore')


@numba.njit()
def find_r200_m200(rel_pos, part_mass, rhom):
    dists = np.sqrt(np.sum(np.square(rel_pos), axis=1))
    dists.sort()
    mass_prof = part_mass * np.arange(1, len(dists)+1)
    loc = np.argmax(mass_prof / (4 / 3 * np.pi * dists ** 3) <= 200 * rhom)
    loc2 = np.argmax(mass_prof / (4 / 3 * np.pi * dists ** 3) <= 5000 * rhom)
    # vel_prof_sq = G_gravity * mass_prof / dists
    argloc = np.argsort(dists)[:loc2]

    # return dists[loc], mass_prof[loc], np.max(vel_prof_sq)
    return dists[loc], mass_prof[loc], dists[loc2], argloc


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
    """Runs the classifier for each seed in a sub-box.
    Additionally, percolates all found haloes by:
        1. Resolves parent-sub halo relationship by only allowing less massive
           structures to orbit more massive ones.
        2. Assigns shared sub-haloes between parents to the closest parent.
        3. Assigns shared particles between parents to the closest parent.
    The distance metric is the phase-space distance:
    \begin{equation*}
    d^{2} = \frac{|\vec{x}_p - \vec{x}_h|^2}{r_{v_\max}^2} + 
                \frac{|\vec{v}_p - \vec{v}_h|^2}{\sigma_v^2}
    \end{equation*}
    where
    \begin{equation*}
    r_{v_\max}^2 = \frac{v_{\max}^2}{\frac{4\pi}{3}G\rho_{200}}
    \end{equation*}
    
    See P. Behroozi (2013) for a discussion on this metric.

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
    save_path = path + 'sub_box_catalogues/'

    # Load seeds
    pos_seed, vel_seed, hid_seed, row_seed = load_seeds(sub_box_id, boxsize,
                                                        subsize, path, padding)
    if any([p is None for p in (pos_seed, vel_seed, hid_seed, row_seed)]):
        return None
    hid_seed_sb = hid_seed

    # n_seeds = len(hid_seed)
    # Load adjacent seeds
    pos_seed_adj, vel_seed_adj, hid_seed_adj, row_seed_adj = load_seeds(
        sub_box_id, boxsize, subsize, path, padding, adjacent=True)
    # Concatenate seeds
    hid_seed = np.hstack([hid_seed, hid_seed_adj])
    hid_seed = np.hstack([hid_seed, hid_seed_adj])
    _, index = np.unique(hid_seed, return_index=True)

    hid_seed = hid_seed[index]
    row_seed = np.hstack([row_seed, row_seed_adj])[index]
    pos_seed = np.vstack([pos_seed, pos_seed_adj])[index]
    vel_seed = np.vstack([vel_seed, vel_seed_adj])[index]
    n_seeds = len(hid_seed)

    # Load particles
    pos_part, vel_part, pid_part, row_part = load_particles(sub_box_id, boxsize,
                                                            subsize, path,
                                                            padding)

    # Create empty catalog of found halos and a dictionary with the PIDs. The
    # temporary objects are to save raw results before percolation.
    halo_members = {}
    halo_members_temp = {}
    halo_subs = {}
    halo_subs_temp = {}

    col_names = ('OHID', 'pos', 'vel', 'R200m', 'M200m', 'Morb')
    haloes = pd.DataFrame(columns=col_names)
    haloes_temp = pd.DataFrame(columns=col_names)

    # Load calibration parameters
    with h5.File(path + 'calibration_pars.hdf5', 'r') as hdf:
        m_pos, b_pos = hdf['pos'][()]
        m_neg, b_neg = hdf['neg'][()]
    pars = (m_pos, b_pos, m_neg, b_neg)

    for i in range(n_seeds):
        # Classify particles ===================================================
        rel_pos = relative_coordinates(pos_seed[i], pos_part, boxsize)
        rel_vel = vel_part - vel_seed[i]
        # r200, m200, vmax = find_r200_m200(rel_pos, part_mass, rhom)
        r200, m200, r5000, argloc = find_r200_m200(rel_pos, part_mass, rhom)
        # Classify
        mask_orb = classify(rel_pos, rel_vel, r200, m200, pars)
        # Compute phase space distance from particle to halo
        # sigmax = vmax**2 / (G_gravity * 200 * rhom * 4 * np.pi / 3)
        sigmax = r5000**2
        # sigmav = np.var(rel_vel)
        sigmav = np.median(np.sum(np.square(rel_vel[argloc]), axis=1))
        dphsq = np.sum(np.square(rel_pos), axis=1) / sigmax + \
            np.sum(np.square(rel_vel), axis=1) / sigmav

        # Ignore seed if it does not have the minimum mass to be considered a
        # halo. Early exit to avoid further computation for a non-halo seed
        if mask_orb.sum() < min_num_part:
            continue

        # Select orbiting particles' PID
        row_idx_order = np.argsort(row_part[mask_orb])
        orb_pid = pid_part[mask_orb][row_idx_order]
        orb_arg = row_part[mask_orb][row_idx_order]
        orb_dph = dphsq[mask_orb][row_idx_order]

        # Append halo to halo catalogue if there are at least min_dm_part
        # orbiting particles
        haloes_temp.loc[len(haloes_temp.index)] = [
            hid_seed[i],
            pos_seed[i],
            vel_seed[i],
            r200,
            m200,
            part_mass * mask_orb.sum(),
        ]
        halo_members_temp[hid_seed[i]] = {'PID': orb_pid,
                                          'row_idx': orb_arg,
                                          'dph': orb_dph,
                                          }

        # Classify seeds =======================================================
        mask_self = hid_seed != hid_seed[i]
        rel_pos = relative_coordinates(
            pos_seed[i], pos_seed[mask_self], boxsize)
        rel_vel = vel_seed[mask_self] - vel_seed[i]
        # Classify
        mask_orb_seed = classify(rel_pos, rel_vel, r200, m200, pars)

        if mask_orb_seed.sum() > 0:
            # Compute phase space distance from particle to halo
            dphsq = np.sum(np.square(rel_pos), axis=1) / sigmax + \
                np.sum(np.square(rel_vel), axis=1) / sigmav

            # Select orbiting particles' PID
            orb_pid_seed = hid_seed[mask_self][mask_orb_seed]
            orb_dph_seed = dphsq[mask_orb_seed]

            halo_subs_temp[hid_seed[i]] = {'OHID': orb_pid_seed,
                                           'dph': orb_dph_seed,
                                           }

    if len(haloes_temp.index) < 1:
        return None

    # Percolate sub-haloes =====================================================

    # Rank order haloes by mass
    haloes_temp.sort_values(by='Morb', ascending=False, inplace=True)
    ohids_sorted = haloes_temp['OHID'].values
    # Select hales with members only
    members_keys = np.array([item for item in halo_subs_temp.keys()])
    mask_with_subs = np.isin(ohids_sorted, members_keys, assume_unique=True)
    ohids_with_subs = ohids_sorted[mask_with_subs]

    # It can happen that two or more haloes are mutually orbiting. However, a
    # more massive halo cannot orbit a smaller one (by definition). Therefore,
    # we rank order all haloes, and remove all orbiting haloes that are more
    # massive than the halo itself.
    temp_subs = {}
    for i in range(1, len(ohids_with_subs)):
        ohid_massive = ohids_with_subs[:i]
        ohid_current = ohids_with_subs[i]
        members_current = halo_subs_temp[ohid_current]['OHID']
        mask = np.isin(members_current, ohid_massive)
        if mask.sum() == 0:
            temp_subs[ohid_current] = {'OHID': members_current,
                                       'dph': halo_subs_temp[ohid_current]['dph'],
                                       }
        else:
            mask2 = np.isin(members_current, ohid_massive, invert=True)
            temp_subs[ohid_current] = {'OHID': members_current[mask2],
                                       'dph': halo_subs_temp[ohid_current]['dph'][mask2],
                                       }
    # Reverse the members dictionary. PID: HID and PID: dph
    members_rev = defaultdict(list)
    dph_rev = defaultdict(list)

    members_keys = np.array([item for item in temp_subs.keys()])
    for hid in members_keys:
        for i, sub_hid in enumerate([*temp_subs[hid]['OHID']]):
            members_rev[sub_hid].append(hid)
            dph_rev[sub_hid].append(temp_subs[hid]['dph'][i])

    # Look for repeated members
    repeated_members = []
    for sub_hid, elements in members_rev.items():
        if len(elements) > 1:
            repeated_members.append(sub_hid)

    # Create a dictionary with the particles to remove per halo. HID: PID
    sub_hids_to_remove = defaultdict(list)
    for sub_hid in repeated_members:
        current_sub_hid = np.array(members_rev[sub_hid])
        current_dph = np.array(dph_rev[sub_hid])
        loc_min = np.argmin(current_dph)
        mask_remove = current_dph != current_dph[loc_min]

        for hid in current_sub_hid[mask_remove]:
            sub_hids_to_remove[hid].append(sub_hid)
    sub_hids_to_remove_keys = np.array(
        [item for item in sub_hids_to_remove.keys()])

    # Create a new members catalogue, removing particles form haloes.
    for hid in members_keys:
        if hid in hid_seed_sb:
            if hid in sub_hids_to_remove_keys:
                pid_remove = sub_hids_to_remove[hid]
                mask_keep = np.isin(
                    temp_subs[hid]['OHID'],
                    pid_remove,
                    assume_unique=True,
                    invert=True
                )
                if mask_keep.sum() == 0:
                    continue
                halo_subs[hid] = {
                    'OHID': temp_subs[hid]['OHID'][mask_keep],
                }
            else:
                halo_subs[hid] = {
                    'OHID': temp_subs[hid]['OHID'],
                }

    # Tag parents and subs
    pids = np.full(
        shape=len(haloes_temp['OHID']), fill_value=-1, dtype=np.int32)
    ohids = haloes_temp['OHID'].values
    for hid, value in halo_subs.items():
        mask = np.isin(ohids, value['OHID'], assume_unique=True)
        pids[mask] = hid

    # Percolate particles ======================================================
    # Ignoring sub-haloes
    ohids_to_skip = haloes_temp['OHID'].values[pids != -1]
    # Reverse the members dictionary. PID: HID and PID: dph
    members_rev = defaultdict(list)
    dph_rev = defaultdict(list)

    members_keys = np.array([item for item in halo_members_temp.keys()])
    for hid in members_keys:
        if hid in ohids_to_skip:
            continue
        for i, sub_hid in enumerate([*halo_members_temp[hid]['PID']]):
            members_rev[sub_hid].append(hid)
            dph_rev[sub_hid].append(halo_members_temp[hid]['dph'][i])

    # Look for repeated members
    repeated_members = []
    for sub_hid, elements in members_rev.items():
        if len(elements) > 1:
            repeated_members.append(sub_hid)

    pids_to_remove = defaultdict(list)
    for pid in repeated_members:
        current_pid = np.array(members_rev[pid])
        current_dph = np.array(dph_rev[pid])
        loc_min = np.argmin(current_dph)
        mask_remove = current_dph != current_dph[loc_min]

        for hid in current_pid[mask_remove]:
            pids_to_remove[hid].append(pid)

    pids_to_remove_keys = pids_to_remove.keys()

    removed_haloes = []
    # Create a new members catalogue, removing particles form haloes.
    for hid in members_keys:
        if hid in hid_seed_sb:
            if hid in pids_to_remove_keys:
                pid_remove = pids_to_remove[int(hid)]
                mask_keep = np.isin(
                    halo_members_temp[hid]['PID'],
                    pid_remove,
                    assume_unique=True,
                    invert=True,
                )
                if mask_keep.sum() < min_num_part:
                    removed_haloes.append(hid)

                halo_members[hid] = {
                    'PID': halo_members_temp[hid]['PID'][mask_keep],
                    'row_idx': halo_members_temp[hid]['row_idx'][mask_keep],
                }
            else:
                halo_members[hid] = {
                    'PID': halo_members_temp[hid]['PID'],
                    'row_idx': halo_members_temp[hid]['row_idx'],
                }

    # Update Morb
    mask_in_sb = np.isin(haloes_temp['OHID'], hid_seed_sb)
    mass_new = np.zeros_like(haloes_temp['Morb'].values[mask_in_sb])
    for i, hid in enumerate(haloes_temp['OHID'].values[mask_in_sb]):
        if hid in removed_haloes:
            continue
        mass_new[i] = part_mass * len(halo_members[hid]['PID'])
    # Select haloes in subbox
    haloes = haloes_temp[mask_in_sb]
    mask_mass = mass_new > 0
    
    # Exit if there are no haloes left
    if mask_mass.sum() < 1:
        return None

    haloes = haloes[mask_mass]
    haloes['Morb'] = mass_new[mask_mass]
    haloes['PID'] = pids[mask_in_sb][mask_mass]

    # Save catalogue ===========================================================
    with h5.File(save_path + f'{sub_box_id}.hdf5', 'w') as hdf:
        # Save halo catalogue
        for i, key in enumerate(haloes.columns):
            if key in ["pos", "vel"]:
                data = np.stack(haloes[key].values)
            else:
                data = haloes[key].values
            hdf.create_dataset(f'halo/{key}', data=data)

        # Save halo members (particles)
        # Dataset names must be strings (thus are not properly sorted)
        for item, _ in halo_members.items():
            hdf.create_dataset(f'members/part/{str(item)}/PID',
                               data=halo_members[item]['PID'])
            hdf.create_dataset(f'members/part/{str(item)}/row_idx',
                               data=halo_members[item]['row_idx'])
        # Save halo members (seed)
        if len(halo_subs.keys()) > 0:
            for item, _ in halo_subs.items():
                hdf.create_dataset(f'members/halo/{str(item)}/OHID',
                                   data=halo_subs[item]['OHID'])

    return None


@timer
def generate_full_box_catalogue(
    path: str,
    min_num_part: int,
    part_mass: float,
    rhom: float,
    boxsize: float,
    subsize: float,
    padding: float = 5.0,
    n_threads: int = None,
    # hdf: bool = False,
) -> None:
    """Generates a halo catalogue using the kinetic mass criterion to classify
    particles into orbiting or infalling.

    Parameters
    ----------
    path : str
        Location from where to load the file
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
    padding : float, optional
        Only particles up to this distance from the sub-box edge are considered 
        for classification. Defaults to 5
    n_threads : int
        Number of threads

    Returns
    -------
    None
    """
    # Create directory if it does not exist
    save_path = path + 'sub_box_catalogues/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    n_sub_boxes = np.int_(np.ceil(boxsize / subsize))**3

    func = partial(classify_seeds_in_sub_box, min_num_part=min_num_part,
                   part_mass=part_mass, rhom=rhom, boxsize=boxsize,
                   subsize=subsize, path=path, padding=padding)

    with Pool(n_threads) as pool:
        list(tqdm(pool.imap(func, range(n_sub_boxes)),
                  total=n_sub_boxes, colour="green", ncols=100,
                  desc='Generating halo catalogue'))

    # Consolidate catalogue
    m200m, morb, ohid, r200m, pos, vel, pid = ([] for _ in range(7))
    files = os.listdir(save_path)
    for f in tqdm(files, ncols=100, desc='Consolidating catalogue', colour='green'):
        with h5.File(save_path+f, 'r') as hdf:
            if 'halo' in hdf.keys():
                m200m.append(hdf['halo/M200m'][()])
                r200m.append(hdf['halo/R200m'][()])
                morb.append(hdf['halo/Morb'][()])
                ohid.append(hdf['halo/OHID'][()])
                pos.append(hdf['halo/pos'][()])
                vel.append(hdf['halo/vel'][()])
                pid.append(hdf['halo/PID'][()])

    ohid = np.concatenate(ohid)
    # NOTE: Redundant?
    _, index = np.unique(ohid, return_index=True)

    with h5.File(path + 'halo_catalogue.hdf5', 'w') as hdf:
        hdf.create_dataset('M200m', data=np.concatenate(m200m)[index])
        hdf.create_dataset('R200m', data=np.concatenate(r200m)[index])
        hdf.create_dataset('OHID', data=ohid[index])
        hdf.create_dataset('PID', data=np.concatenate(pid)[index])
        hdf.create_dataset('Morb', data=np.concatenate(morb)[index])
        hdf.create_dataset('pos', data=np.concatenate(pos)[index])
        hdf.create_dataset('vel', data=np.concatenate(vel)[index])

    # Consolidate members catalogue
    with h5.File(path + 'halo_members.hdf5', 'w') as hdf:
        for f in tqdm(files, ncols=100, desc='Consolidating members', colour='green'):
            with h5.File(save_path+f, 'r') as hdf_load:
                if 'members' in hdf_load.keys():
                    for hid in hdf_load['members/part'].keys():
                        hdf.create_dataset(
                            f'{hid}/PID', data=hdf_load[f'members/part/{hid}/PID'][()])
                        hdf.create_dataset(
                            f'{hid}/row_idx', data=hdf_load[f'members/part/{hid}/row_idx'][()])

    return


if __name__ == "__main__":
    pass
