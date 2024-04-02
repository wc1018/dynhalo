import os
import pickle
from collections import defaultdict
from functools import partial
from itertools import repeat
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


@numba.njit()
def find_r200_m200_new(rel_pos, part_mass, rhom):
    dists = np.sqrt(np.sum(np.square(rel_pos), axis=1))
    dists.sort()
    mass_prof = part_mass * np.arange(1, len(dists)+1)
    loc = np.argmax(mass_prof / (4 / 3 * np.pi * dists ** 3) <= 200 * rhom)
    vel_prof_sq = G_gravity * mass_prof / dists

    return dists[loc], mass_prof[loc], np.max(vel_prof_sq)


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
        # r200, m200 = find_r200_m200(rel_pos, part_mass, rhom)
        r200, m200, vmax = find_r200_m200_new(rel_pos, part_mass, rhom)
        v200sq = G_gravity * m200 / r200
        rvmaxsq = vmax**2 / (G_gravity * 200 * rhom * 4 * np.pi / 3)
        # Classify
        mask_orb = classify(rel_pos, rel_vel, r200, m200, pars)
        # Compute phase space distance from particle to halo
        dphsq = np.sum(np.square(rel_pos), axis=1) / r200**2 + \
            np.sum(np.square(rel_vel), axis=1) / v200sq
        dphsq2 = np.sum(np.square(rel_pos), axis=1) / rvmaxsq + \
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
        orb_dph2 = dphsq2[mask_orb][row_idx_order]

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
                                     'dph': orb_dph,
                                     'dph2': orb_dph2}

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
            dphsq2 = np.sum(np.square(rel_pos), axis=1) / rvmaxsq + \
                np.sum(np.square(rel_vel), axis=1) / v200sq
            # Select orbiting particles' PID
            row_idx_order = np.argsort(row_seed_adj[mask_self][mask_orb_seed])
            orb_pid_seed = hid_seed_adj[mask_self][mask_orb_seed][row_idx_order]
            orb_arg_seed = row_seed_adj[mask_self][mask_orb_seed][row_idx_order]
            orb_dph_seed = dphsq[mask_orb_seed][row_idx_order]
            orb_dph2_seed = dphsq2[mask_orb_seed][row_idx_order]

            halo_subs[hid_seed[i]] = {'OHID': orb_pid_seed,
                                      'row_idx': orb_arg_seed,
                                      'dph': orb_dph_seed,
                                      'dph2': orb_dph2_seed}

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
            hdf.create_dataset(f'members/part/{str(item)}/dph2',
                               data=halo_members[item]['dph2'])
        # Save halo members (seed)
        if len(halo_subs.keys()) > 0:
            for item, _ in halo_subs.items():
                hdf.create_dataset(f'members/halo/{str(item)}/OHID',
                                   data=halo_subs[item]['OHID'])
                hdf.create_dataset(f'members/halo/{str(item)}/row_idx',
                                   data=halo_subs[item]['row_idx'])
                hdf.create_dataset(f'members/halo/{str(item)}/dph',
                                   data=halo_subs[item]['dph'])
                hdf.create_dataset(f'members/halo/{str(item)}/dph2',
                                   data=halo_subs[item]['dph2'])

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
    hdf: bool = False,
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

    with Pool(n_threads) as pool:
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
                members[int(hid)] = {
                    'PID': hdf[f'members/part/{hid}/PID'][()],
                    'dph': hdf[f'members/part/{hid}/dph'][()],
                    'dph2': hdf[f'members/part/{hid}/dph2'][()],
                    'row_idx': hdf[f'members/part/{hid}/row_idx'][()],
                }

            if 'halo' in hdf['members'].keys():
                for hid in hdf['members/halo'].keys():
                    members_seed[int(hid)] = {
                        'OHID': hdf[f'members/halo/{hid}/OHID'][()],
                        'dph': hdf[f'members/halo/{hid}/dph'][()],
                        'dph2': hdf[f'members/halo/{hid}/dph2'][()],
                        'row_idx': hdf[f'members/halo/{hid}/row_idx'][()],
                    }

    with h5.File(path + 'dynamical_halo_catalogue.hdf5', 'w') as hdf:
        hdf.create_dataset('M200m_all', data=np.concatenate(m200_all))
        hdf.create_dataset('Morb', data=np.concatenate(morb))
        hdf.create_dataset('OHID', data=np.concatenate(ohid))
        hdf.create_dataset('R200m', data=np.concatenate(r200m))
        hdf.create_dataset('pos', data=np.concatenate(pos))
        hdf.create_dataset('vel', data=np.concatenate(vel))

    # Save pickle objects for fast loading.
    if not os.path.isdir(path + 'pickle/'):
        os.mkdir(path + 'pickle/')

    with open(path + 'pickle/dynamical_halo_members.pickle', 'wb') as handle:
        pickle.dump(members, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(path + 'pickle/dynamical_halo_members_sub_haloes.pickle', 'wb') as handle:
        pickle.dump(members_seed, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if hdf:
        if not os.path.isdir(path + 'hdf/'):
            os.mkdir(path + 'hdf/')

        with h5.File(path + 'hdf/dynamical_halo_members_sub_haloes.hdf5', 'w') as hdf:
            for hid in tqdm(members_seed.keys(), ncols=100, desc='Saving subhaloes', colour='green'):
                hdf.create_dataset(
                    f'{hid}/OHID', data=members_seed[hid]['OHID'])
                hdf.create_dataset(f'{hid}/dph', data=members_seed[hid]['dph'])
                hdf.create_dataset(
                    f'{hid}/dph2', data=members_seed[hid]['dph2'])
                hdf.create_dataset(
                    f'{hid}/row_idx', data=members_seed[hid]['row_idx'])

        with h5.File(path + 'hdf/dynamical_halo_members.hdf5', 'w') as hdf:
            for hid in tqdm(members.keys(), ncols=100, desc='Saving members', colour='green'):
                hdf.create_dataset(f'{hid}/PID', data=members[hid]['PID'])
                hdf.create_dataset(f'{hid}/dph', data=members[hid]['dph'])
                hdf.create_dataset(f'{hid}/dph2', data=members[hid]['dph2'])
                hdf.create_dataset(
                    f'{hid}/row_idx', data=members[hid]['row_idx'])

    return None


@timer
def percolate_sub_haloes(
    path: str,
    n_threads: int = None,
) -> None:
    """_summary_

    Parameters
    ----------
    path : str
        _description_
    """
    # First, load the catalogue and remove duplicates if any
    with h5.File(path + 'dynamical_halo_catalogue.hdf5', 'r') as hdf:
        ohid = hdf['OHID'][()]
        morb = hdf['Morb'][()]
        pos = hdf['pos'][()]
        r200 = hdf['R200m'][()]
    # Get unique IDs
    _, index = np.unique(ohid, return_index=True)
    # Rank order them by mass.
    order = np.argsort(morb[index])[::-1]
    # Apply masks
    morb = morb[index][order]
    ohid = ohid[index][order]
    pos = pos[index][order]
    r200 = r200[index][order]

    # Save to file
    with h5.File(path + 'dynamical_halo_catalogue_no_duplicated.hdf5', 'w') as hdf_save, \
            h5.File(path + 'dynamical_halo_catalogue.hdf5', 'r') as hdf:
        for feature in hdf.keys():
            hdf_save.create_dataset(feature, data=hdf[feature][()][index])

    # Now, load the sub-haloes
    with open(path + 'pickle/dynamical_halo_members_sub_haloes.pickle', 'rb') as handle:
        members = pickle.load(handle)
    # Create a dictionary with integer keys.
    # members = {int(item): value for item, value in members.items()}
    members_keys = np.array([item for item in members.keys()])

    # Select haloes with sub-haloes only. They should exactly match the members
    # list
    mask_with_subs = np.isin(ohid, members_keys, assume_unique=True)
    ohids_with_subs = ohid[mask_with_subs]

    # It can happen that two or more haloes are mutually orbiting. However, a
    # more massive halo cannot orbit a smaller one (definition). Therefore,
    # we rank order all haloes, and remove all orbiting haloes that are more
    # massive than the halo itself.
    fname = path + 'pickle/dynamical_halo_members_sub_haloes_clean.pickle'
    if os.path.exists(fname):
        with open(fname, 'rb') as handle:
            members_ro = pickle.load(handle)
    else:
        n_haloes = len(ohids_with_subs)

        global remove_massive

        def remove_massive(i):
            ohid_massive = ohids_with_subs[:i]
            ohid_current = ohids_with_subs[i]
            members_current = members[ohid_current]['OHID']
            mask = np.isin(members_current, ohid_massive, invert=True)
            if mask.sum() > 1:
                return ohid_current, {'OHID': members_current[mask],
                                      'row_idx': members[ohid_current]['row_idx'][mask],
                                      'dph': members[ohid_current]['dph'][mask],
                                      'dph2': members[ohid_current]['dph2'][mask]}
            else:
                return ohid_current, None

        with Pool(n_threads) as pool:
            members_ro = list(
                tqdm(
                    pool.map(
                        remove_massive,
                        np.arange(n_haloes),
                    ),
                    ncols=100,
                    desc='Rank-order haloes',
                    colour='blue',
                    total=n_haloes
                )
            )
        del remove_massive
        members_ro = {item[0]: item[1]
                      for item in members_ro if item[1] is not None}
        with open(fname, 'wb') as handle:
            pickle.dump(members_ro, handle, protocol=pickle.HIGHEST_PROTOCOL)

    members_ro_keys = np.array([item for item in members_ro.keys()])

    # Reverse the members dictionary. PID: HID and PID: dph
    members_rev = defaultdict(list)
    dph_rev = defaultdict(list)
    dph2_rev = defaultdict(list)

    for hid in tqdm(members_ro_keys, ncols=100, desc='Reversing dicts', colour='blue'):
        for i, sub_hid in enumerate([*members_ro[hid]['OHID']]):
            members_rev[sub_hid].append(hid)
            dph_rev[sub_hid].append(members_ro[hid]['dph'][i])
            dph2_rev[sub_hid].append(members_ro[hid]['dph2'][i])

    # Look for repeated members
    repeated_members = []
    for sub_hid, elements in tqdm(members_rev.items(), ncols=100,
                                  desc='Looking for repetitions', colour='blue'):
        if len(elements) > 1:
            repeated_members.append(sub_hid)

    # Remove with `dph`
    distances = ((dph_rev, ''), (dph2_rev, '_2'))

    for item in distances:
        dph_item, suffix = item

        # Create a dictionary with the particles to remove per halo. HID: PID
        sub_hids_to_remove = defaultdict(list)
        for sub_hid in tqdm(repeated_members, ncols=100, desc='Selecting OHIDs', colour='blue'):
            current_sub_hid = np.array(members_rev[sub_hid])
            current_dph = np.array(dph_item[sub_hid])
            loc_min = np.argmin(current_dph)
            mask_remove = current_dph != current_dph[loc_min]

            for hid in current_sub_hid[mask_remove]:
                sub_hids_to_remove[hid].append(sub_hid)
        sub_hids_to_remove_keys = sub_hids_to_remove.keys()

        # Create a new members catalogue, removing particles form haloes.
        new_members = {}
        for hid in tqdm(members_ro_keys, ncols=100, desc='Removing members', colour='blue'):
            if hid in sub_hids_to_remove_keys:
                pid_remove = sub_hids_to_remove[hid]
                mask_keep = np.isin(
                    members_ro[hid]['OHID'],
                    pid_remove,
                    assume_unique=True,
                    invert=True
                )
                if mask_keep.sum() == 0:
                    continue
                new_members[hid] = {
                    'OHID': members_ro[hid]['OHID'][mask_keep],
                    'row_idx': members_ro[hid]['row_idx'][mask_keep],
                }
            else:
                new_members[hid] = {
                    'OHID': members_ro[hid]['OHID'],
                    'row_idx': members_ro[hid]['row_idx'],
                }

        with open(path + f'pickle/dynamical_halo_members_sub_haloes_percolated{suffix}.pickle', 'wb') as handle:
            pickle.dump(new_members, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Tag subs with parent halo ID.
        with h5.File(path + 'dynamical_halo_catalogue_no_duplicated.hdf5', 'r') as hdf, \
                h5.File(path + f'dynamical_halo_catalogue_no_duplicated_subs_perc{suffix}.hdf5', 'w') as hdf_save:

            pids = np.full(shape=len(hdf['OHID']),
                           fill_value=-1, dtype=np.int32)
            ohids = hdf['OHID'][()]
            for hid, value in tqdm(new_members.items(), ncols=100, desc='Finding parents', colour='blue'):
                mask = np.isin(ohids, value['OHID'], assume_unique=True)
                pids[mask] = hid

            for feature in hdf.keys():
                hdf_save.create_dataset(feature, data=hdf[feature][()])
            hdf_save.create_dataset('PID', data=pids)

    return


@timer
def percolate_members(path: str, min_num_part: int, hdf: bool = True) -> None:
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
    with h5.File(path + 'dynamical_halo_catalogue_no_duplicated_subs_perc.hdf5', 'r') as hdf:
        ohids = hdf['OHID'][()]
        mass = hdf['Morb'][()]
        pid = hdf['PID'][()]
    mask_parents = (pid == -1)
    mask_subs = (pid != -1)
    ohids_to_skip = ohids[mask_subs]
    ohids = ohids[mask_parents]
    mass = mass[mask_parents]

    with open(path + 'pickle/dynamical_halo_members.pickle', 'rb') as handle:
        members = pickle.load(handle)
    members_keys = np.array([item for item in members.keys()])

    members_rev = defaultdict(list)
    dph_rev = defaultdict(list)
    dph2_rev = defaultdict(list)

    for hid in tqdm(ohids, ncols=100, desc='Reversing dicts', colour='blue'):
        if hid in ohids_to_skip:
            continue
        for i, sub_hid in enumerate(members[hid]['PID']):
            members_rev[sub_hid].append(hid)
            dph_rev[sub_hid].append(members[hid]['dph'][i])
            dph2_rev[sub_hid].append(members[hid]['dph2'][i])

    # Look for repeated members
    repeated_members = []
    for sub_hid, elements in tqdm(members_rev.items(), ncols=100,
                                  desc='Looking for repetitions', colour='blue'):
        if len(elements) > 1:
            repeated_members.append(sub_hid)

    # Remove with `dph`
    distances = ((dph_rev, ''), (dph2_rev, '_2'))
    for item in distances:
        dph_item, suffix = item

        # Create a dictionary with the particles to remove per halo. HID: PID
        pids_to_remove = defaultdict(list)
        for pid in tqdm(repeated_members, ncols=100, desc='Selecting PIDs', colour='blue'):
            current_pid = np.array(members_rev[pid])
            current_dph = np.array(dph_item[pid])
            loc_min = np.argmin(current_dph)
            mask_remove = current_dph != current_dph[loc_min]

            for hid in current_pid[mask_remove]:
                pids_to_remove[hid].append(pid)

        pids_to_remove_keys = pids_to_remove.keys()

        # Create a new members catalogue, removing particles form haloes.
        new_members = {}
        for hid in tqdm(members_keys, ncols=100, desc='Removing members', colour='blue'):
            if hid in pids_to_remove_keys:
                pid_remove = pids_to_remove[hid]
                mask_keep = np.isin(
                    members[hid]['PID'],
                    pid_remove,
                    assume_unique=True,
                    invert=True,
                )
                if mask_keep.sum() < min_num_part:
                    continue

                new_members[hid] = {
                    'PID': members[hid]['PID'][mask_keep],
                    'row_idx': members[hid]['row_idx'][mask_keep],
                }
            else:
                new_members[hid] = {
                    'PID': members[hid]['PID'],
                    'row_idx': members[hid]['row_idx'],
                }

        # Save new members catalogue
        with open(path + f'pickle/dynamical_halo_members_percolated{suffix}.pickle', 'wb') as handle:
            pickle.dump(new_members, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if hdf:
            new_members_keys = new_members.keys()
            with h5.File(path + f'hdf/dynamical_halo_members_percolated{suffix}.hdf5', 'w') as hdf:
                for hid in tqdm(new_members_keys, ncols=100, desc='Saving members', colour='blue'):
                    hdf.create_dataset(
                        f'{hid}/PID', data=new_members[hid]['PID'])
                    hdf.create_dataset(
                        f'{hid}/row_idx', data=new_members[hid]['row_idx'])

    return None


def percolated_mass_catalogue(path: str, part_mass: float, min_num_part: int) -> None:

    with open(path + 'pickle/dynamical_halo_members_percolated.pickle', 'rb') as handle:
        members = pickle.load(handle)
    members_keys = np.array([item for item in members.keys()])

    with h5.File(path + 'dynamical_halo_catalogue_no_duplicated_subs_perc.hdf5', 'r') as hdf:
        ohids = hdf['OHID'][()]
        mass_old = hdf['Morb'][()]

    mass = np.zeros_like(mass_old)
    for i, hid in enumerate(tqdm(ohids, ncols=100, desc='Calculating masses', colour='blue')):
        if hid in members_keys:
            n_memb = len(members[hid]['PID'])
            mass[i] = part_mass * n_memb
        else:
            mass[i] = 0
    min_mass = min_num_part * part_mass
    mask_mass = (mass > min_mass)
    rh = 0.8403 * np.power(mass / 1e14, 0.226)

    with h5.File(path + 'dynamical_halo_catalogue_no_duplicated_subs_perc.hdf5', 'r') as hdf, \
            h5.File(path + 'dynamical_halo_catalogue_percolated.hdf5', 'w') as hdf_save:
        for feature, value in hdf.items():
            if feature == 'Morb':
                continue
            else:
                hdf_save.create_dataset(feature, data=value[()][mask_mass])
        hdf_save.create_dataset('Morb', data=mass[mask_mass])
        hdf_save.create_dataset('Rh', data=rh[mask_mass], dtype=np.float32)

    return


if __name__ == "__main__":
    pass
