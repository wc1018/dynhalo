import os
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from typing import List, Tuple, Union
from warnings import filterwarnings

import h5py as h5
import numpy as np
import pandas as pd
from tqdm import tqdm

from dynhalo.finder.coordinates import relative_coordinates
from dynhalo.finder.subbox import load_particles, load_seeds
from dynhalo.utils import G_gravity, timer

filterwarnings('ignore')


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
    for i, _ in enumerate(dists):
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
    save_path = path + 'sub_box_catalogues/'

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
    vel_seed_adj = np.vstack([vel_seed, vel_seed_adj])

    # Load particles
    pos_part, vel_part, pid_part, row_part = load_particles(sub_box_id, boxsize,
                                                            subsize, path,
                                                            padding)

    # Create empty catalog of found halos and a dictionary with the PIDs
    halo_members = {}
    halo_subs = {}
    col_names = ("OHID", "pos", "vel", "R200m", "M200m_all", "Morb")
    haloes = pd.DataFrame(columns=col_names)

    # Load calibration parameters
    with h5.File(path + 'calibration_pars.hdf5', 'r') as hdf:
        m_pos, b_pos = hdf['pos'][()]
        m_neg, b_neg = hdf['neg'][()]
    pars = (m_pos, b_pos, m_neg, b_neg)

    for i in range(n_seeds):
        # Classify particles ===================================================
        rel_pos = relative_coordinates(pos_seed[i], pos_part, boxsize)
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
        row_idx_order = np.argsort(row_part[mask_orb])
        orb_pid = pid_part[mask_orb][row_idx_order]
        orb_arg = row_part[mask_orb][row_idx_order]
        orb_dph = dphsq[mask_orb][row_idx_order]

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
        halo_members[hid_seed[i]] = {'PID': orb_pid,
                                     'row_idx': orb_arg,
                                     'dph': orb_dph}

        # Classify seeds =======================================================
        mask_self = hid_seed_adj != hid_seed[i]
        rel_pos = relative_coordinates(
            pos_seed[i], pos_seed_adj[mask_self], boxsize)
        rel_vel = vel_seed_adj[mask_self] - vel_seed[i]
        # Classify
        mask_orb_seed = classify(rel_pos, rel_vel, r200, m200, pars)

        if mask_orb_seed.sum() > 0:
            # Compute phase space distance from particle to halo
            dphsq = np.sum(np.square(rel_pos), axis=1) / r200**2 + \
                np.sum(np.square(rel_vel), axis=1) / v200sq
            # Select orbiting particles' PID
            row_idx_order = np.argsort(row_seed_adj[mask_self][mask_orb_seed])
            orb_pid_seed = hid_seed_adj[mask_self][mask_orb_seed][row_idx_order]
            orb_arg_seed = row_seed_adj[mask_self][mask_orb_seed][row_idx_order]
            orb_dph_seed = dphsq[mask_orb_seed][row_idx_order]

            halo_subs[hid_seed[i]] = {'OHID': orb_pid_seed,
                                      'row_idx': orb_arg_seed,
                                      'dph': orb_dph_seed}

    # Save catalogue
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
            hdf.create_dataset(f'members/part/{str(item)}/dph',
                               data=halo_members[item]['dph'])
        # Save halo members (seed)
        if len(halo_subs.keys()) > 0:
            for item, _ in halo_subs.items():
                hdf.create_dataset(f'members/halo/{str(item)}/OHID',
                                   data=halo_subs[item]['OHID'])
                hdf.create_dataset(f'members/halo/{str(item)}/row_idx',
                                   data=halo_subs[item]['row_idx'])
                hdf.create_dataset(f'members/halo/{str(item)}/dph',
                                   data=halo_subs[item]['dph'])

    return None


@timer
def generate_full_box_catalogue(
    path: str,
    # n_threads: int,
    min_num_part: int,
    part_mass: float,
    rhom: float,
    boxsize: float,
    subsize: float,
    padding: float = 5.0,
) -> None:
    """Generates a halo catalogue using the kinetic mass criterion to classify
    particles into orbiting or infalling.

    Parameters
    ----------
    path : str
        Location from where to load the file
    n_threads : int
        Number of threads
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

    with Pool() as pool:
        list(tqdm(pool.imap(func, range(n_sub_boxes)),
                  total=n_sub_boxes, colour="green", ncols=100,
                  desc='Generating halo catalogue'))

    # Consolidate catalogue
    m200_all, morb, ohid, r200m, pos, vel = ([] for _ in range(6))
    members = {}
    members_seed = {}
    for i in tqdm(range(n_sub_boxes), ncols=100, desc='Reading data', colour='blue'):
        with h5.File(save_path + f'{i}.hdf5', 'r') as hdf:
            m200_all.append(hdf['halo/M200m_all'][()])
            morb.append(hdf['halo/Morb'][()])
            ohid.append(hdf['halo/OHID'][()])
            r200m.append(hdf['halo/R200m'][()])
            pos.append(hdf['halo/pos'][()])
            vel.append(hdf['halo/vel'][()])
            for hid in hdf['members/part'].keys():
                members[hid] = {
                    'PID': hdf[f'members/part/{hid}/PID'][()],
                    'dph': hdf[f'members/part/{hid}/dph'][()],
                    'row_idx': hdf[f'members/part/{hid}/row_idx'][()],
                }
            if 'halo' in hdf['members'].keys():
                for hid in hdf['members/halo'].keys():
                    members_seed[hid] = {
                        'OHID': hdf[f'members/halo/{hid}/OHID'][()],
                        'dph': hdf[f'members/halo/{hid}/dph'][()],
                        'row_idx': hdf[f'members/halo/{hid}/row_idx'][()],
                    }

    with h5.File(path + 'dynamical_halo_catalogue.hdf5', 'w') as hdf:
        hdf.create_dataset('M200m_all', data=np.concatenate(m200_all))
        hdf.create_dataset('Morb', data=np.concatenate(morb))
        hdf.create_dataset('OHID', data=np.concatenate(ohid))
        hdf.create_dataset('R200m', data=np.concatenate(r200m))
        hdf.create_dataset('pos', data=np.concatenate(pos))
        hdf.create_dataset('vel', data=np.concatenate(vel))

    with h5.File(path + 'dynamical_halo_members.hdf5', 'w') as hdf:
        for hid in tqdm(members.keys(), ncols=100, desc='Saving members', colour='green'):
            hdf.create_dataset(f'{hid}/PID', data=members[hid]['PID'])
            hdf.create_dataset(f'{hid}/dph', data=members[hid]['dph'])
            hdf.create_dataset(f'{hid}/row_idx', data=members[hid]['row_idx'])

    with h5.File(path + 'dynamical_halo_members_sub_haloes.hdf5', 'w') as hdf:
        for hid in tqdm(members_seed.keys(), ncols=100, desc='Saving subhaloes', colour='green'):
            hdf.create_dataset(f'{hid}/OHID', data=members_seed[hid]['OHID'])
            hdf.create_dataset(f'{hid}/dph', data=members_seed[hid]['dph'])
            hdf.create_dataset(
                f'{hid}/row_idx', data=members_seed[hid]['row_idx'])

    return None


def percolate_sub_haloes(path: str) -> None:
    """Percolates the sub-haloes. If any halo has membership to more than one
    halo, it is kept in the closest halo and removed from the others. The 
    measure of distance is phase-space distance:

            d^{2}_{p,h} = |x-x_h|^2 / R200^2 + |v-v_h|^2 / V200^2

    Parameters
    ----------
    path : str
        Location from where to load the file

    Returns
    -------
    None
    """
    # Load halo members. HID: OPID, dph, row_idx
    members = {}
    with h5.File(path + 'dynamical_halo_members_sub_haloes.hdf5', 'r') as hdf:
        for hid in tqdm(hdf.keys(), ncols=100, desc='Reading data', colour='blue'):
            members[int(hid)] = {
                'OHID': hdf[f'{hid}/OHID'][()],
                'dph': hdf[f'{hid}/dph'][()],
                'row_idx': hdf[f'{hid}/row_idx'][()]
            }
    members_hids = members.keys()

    # Reverse the members dictionary. PID: HID and PID: dph
    reversed_members = defaultdict(list)
    reversed_members_dph = defaultdict(list)

    for key in tqdm(members_hids, ncols=100, desc='Reversing dicts', colour='blue'):
        for i, item in enumerate(members[key]['OHID']):
            reversed_members[item].append(key)
            reversed_members_dph[item].append(members[key]['dph'][i])

    # Look for repeated members
    repeated_members = []
    for key, item in tqdm(reversed_members.items(), ncols=100,
                          desc='Looking for repetitions', colour='blue'):
        if len(item) > 1:
            repeated_members.append(key)

    # Create a dictionary with the particles to remove per halo. HID: PID
    pids_to_remove = defaultdict(list)
    for item in tqdm(repeated_members, ncols=100, desc='Selecting OHIDs', colour='blue'):
        current_pid = np.array(reversed_members[item])
        current_dph = np.array(reversed_members_dph[item])
        loc_min = np.argmin(current_dph)
        mask_remove = current_dph != current_dph[loc_min]

        for hid in current_pid[mask_remove]:
            pids_to_remove[hid].append(item)
    hids_to_remove = pids_to_remove.keys()

    # Create a new members catalogue, removing particles form haloes.
    new_members = {}
    for key in tqdm(members_hids, ncols=100, desc='Removing members', colour='blue'):
        if key in hids_to_remove:
            pid_remove = pids_to_remove[key]
            mask_keep = ~np.isin(
                members[key]['OHID'], pid_remove, assume_unique=True)
            if mask_keep.sum() == 0:
                continue
            new_members[key] = {
                'OHID': members[key]['OHID'][mask_keep],
                'row_idx': members[key]['row_idx'][mask_keep],
            }
        else:
            new_members[key] = {
                'OHID': members[key]['OHID'],
                'row_idx': members[key]['row_idx'],
            }

    # Save new members catalogue
    new_members_hids = new_members.keys()
    with h5.File(path + 'dynamical_halo_members_sub_haloes_percolated.hdf5', 'w') as hdf:
        for hid in tqdm(new_members_hids, ncols=100, desc='Saving members', colour='blue'):
            # If the HID is a member of another, then it is not a parent halo
            hdf.create_dataset(f'{hid}/OHID', data=new_members[hid]['OHID'])
            hdf.create_dataset(
                f'{hid}/row_idx', data=new_members[hid]['row_idx'])

    return None


@timer
def percolate_members(path: str, part_mass: float, min_num_part: int) -> None:
    """Percolates the halo catalogue. If any particle has membership to more 
    than one halo, it is kept in the closest halo and removed from the others.
    The measure of distance is phase-space distance:

            d^{2}_{p,h} = |x-x_h|^2 / R200^2 + |v-v_h|^2 / V200^2

    Parameters
    ----------
    path : str
        Location from where to load the file
    part_mass : float
        Mass per particle
    """
    # Load halo members. HID: PID, dph, row_idx
    members = {}
    with h5.File(path + 'dynamical_halo_members.hdf5', 'r') as hdf:
        for hid in tqdm(hdf.keys(), ncols=100, desc='Reading data', colour='blue'):
            members[int(hid)] = {
                'PID': hdf[f'{hid}/PID'][()],
                'dph': hdf[f'{hid}/dph'][()],
                'row_idx': hdf[f'{hid}/row_idx'][()]
            }

    # Reverse the members dictionary. PID: HID and PID: dph
    reversed_members = defaultdict(list)
    reversed_members_dph = defaultdict(list)

    for key in tqdm(members.keys(), ncols=100, desc='Reversing dicts', colour='blue'):
        for i, item in enumerate(members[key]['PID']):
            reversed_members[item].append(key)
            reversed_members_dph[item].append(members[key]['dph'][i])

    # Look for repeated members
    repeated_members = []
    for key, item in reversed_members.items():
        if len(item) > 1:
            repeated_members.append(key)

    # Create a dictionary with the particles to remove per halo. HID: PID
    pids_to_remove = defaultdict(list)
    for item in tqdm(repeated_members, ncols=100, desc='Selecting PIDs', colour='blue'):
        current_pid = np.array(reversed_members[item])
        current_dph = np.array(reversed_members_dph[item])
        loc_min = np.argmin(current_dph)
        mask_remove = current_dph != current_dph[loc_min]

        for hid in current_pid[mask_remove]:
            pids_to_remove[hid].append(item)

    # Create a new members catalogue, removing particles form haloes.
    new_members = {}
    for key in tqdm(members.keys(), ncols=100, desc='Removing members', colour='blue'):
        if key in pids_to_remove.keys():
            pid_remove = pids_to_remove[key]
            mask_keep = ~np.isin(
                members[key]['PID'], pid_remove, assume_unique=True)
            new_members[key] = {
                'PID': members[key]['PID'][mask_keep],
                'row_idx': members[key]['row_idx'][mask_keep],
            }
        else:
            new_members[key] = {
                'PID': members[key]['PID'],
                'row_idx': members[key]['row_idx'],
            }

    # Save new members catalogue
    with h5.File(path + 'dynamical_halo_members_percolated.hdf5', 'w') as hdf:
        for hid in tqdm(new_members.keys(), ncols=100, desc='Saving members', colour='blue'):
            if len(members[hid]['PID']) < min_num_part:
                continue
            hdf.create_dataset(f'{hid}/PID', data=new_members[hid]['PID'])
            hdf.create_dataset(
                f'{hid}/row_idx', data=new_members[hid]['row_idx'])

    morb_new = []
    for i, hid in enumerate(tqdm(new_members[hid].keys(), ncols=100,
                                 desc='Saving catalogue', colour='blue')):
        n_memb = len(members[hid]['PID'])
        if n_memb < min_num_part:
            continue
        morb_new.append(part_mass * n_memb)
    morb_new = np.concatenate(morb_new)
    rh = 0.8403 * (morb_new/1e14)**0.226

    with h5.File(path + 'dynamical_halo_catalogue.hdf5', 'r') as hdf, \
            h5.File(path + 'dynamical_halo_catalogue_percolated.hdf5', 'w') as hdf_save:
        for item, _ in hdf.items():
            if item == 'Morb':
                continue
            hdf_save.create_dataset(item, data=hdf[item])
        hdf_save.create_dataset('Morb', data=morb_new)
        hdf_save.create_dataset('Rh_salazar', data=rh, dtype=np.float32)

    return None


def percolate_members_most_massive(path: str, part_mass: float, min_num_part: int) -> None:
    members = {}
    with h5.File(path + 'dynamical_halo_members.hdf5', 'r') as hdf:
        for hid in tqdm(hdf.keys(), ncols=100, desc='Reading data', colour='blue'):
            members[int(hid)] = {
                'PID': hdf[f'{hid}/PID'][()],
                'row_idx': hdf[f'{hid}/row_idx'][()]
            }

    with h5.File(path + 'dynamical_halo_catalogue.hdf5', 'r') as hdf:
        morb = hdf['Morb'][()]
        ohid = hdf['OHID'][()]
    order = np.argsort(morb)
    morb = morb[order]
    ohid = ohid[order]

    for i, hid_1 in enumerate(tqdm(ohid), ncols=100, desc='Percolating haloes', colour='blue'):
        current_pids = members[hid_1]['PID']

        for hid_2 in ohid[i+1:]:
            other_pids = members[hid_2]['PID']
            other_rows = members[hid_2]['row_idx']
            mask_remove = np.isin(other_pids, current_pids, assume_unique=True)
            if mask_remove.sum() > 0:
                members[hid_2]['PID'] = other_pids[~mask_remove]
                members[hid_2]['row_idx'] = other_rows[~mask_remove]

    morb_new = []
    for i, hid in enumerate(tqdm(members[hid].keys(), ncols=100,
                                 desc='Saving catalogue', colour='blue')):
        n_memb = len(members[hid]['PID'])
        if n_memb < min_num_part:
            continue
        morb_new.append(part_mass * n_memb)
    morb_new = np.concatenate(morb_new)
    rh = 0.8403 * (morb_new/1e14)**0.226

    with h5.File(path + 'dynamical_halo_catalogue.hdf5', 'r') as hdf, \
            h5.File(path + 'dynamical_halo_catalogue_percolated_massive.hdf5', 'w') as hdf_save:
        for item, _ in hdf.items():
            if item == 'Morb':
                continue
            hdf_save.create_dataset(item, data=hdf[item])
        hdf_save.create_dataset('Morb', data=morb_new)
        hdf_save.create_dataset('Rh_salazar', data=rh, dtype=np.float32)

    return


if __name__ == "__main__":
    pass
