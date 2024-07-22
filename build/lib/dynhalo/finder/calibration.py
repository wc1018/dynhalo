import os
from typing import Tuple

import h5py as h5
import numpy as np
from scipy.optimize import curve_fit, minimize
from tqdm import tqdm

from dynhalo.finder.coordinates import (get_vr_vt_from_coordinates,
                                        relative_coordinates)
from dynhalo.finder.catalogue import find_r200_m200
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
        Mass density of the universe

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
    for sub_box_id in tqdm(unique_sub_box_ids, desc='Processing sub-box',
                        colour='blue', ncols=100):
        pos, vel, _, _ = load_particles(sub_box_id, boxsize, subsize, path)

        # Iterate over seeds in current sub-box ID
        mask_seeds_in_sub_box = seed_sub_box_id == sub_box_id
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
            r200, m200, _, _ = find_r200_m200(rel_pos, part_mass, rhom)
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
        Mass density of the universe

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


def cost_percentile(b: float, *data) -> float:
    """Cost function for y-intercept b parameter. The optimal value of b is such
    that the `target` percentile of paricles is below the line.

    Parameters
    ----------
    b : float
        Fit parameter
    *data : tuple
        A tuple with `[r, lnv2, slope, target]`, where `slope` is the slope of 
        the line and is fixed, and `target` is the desired percentile

    Returns
    -------
    float
    """
    r, lnv2, slope, target = data
    below_line = (lnv2 < (slope * r + b)).sum()
    return np.log((target - below_line / r.shape[0]) ** 2)


def cost_perp_distance(b: float, *data) -> float:
    """Cost function for y-intercept b parameter. The optimal value of b is such
    that the perpendicular distance of all points to the line is maximal
    Parameters
    ----------
    b : float
        Fit parameter
    *data: tuple
        A tuple with `[r, lnv2, slope, width]`, where `slope` is the slope of 
        the line and is fixed, and `width` is the width of a band around the 
        line within which the distance is computed
        
    Returns
    -------
    float
        _description_
    """
    r, lnv2, slope, width = data
    d = np.abs(lnv2 - slope * r - b) / np.sqrt(1 + slope**2)
    return -np.log(np.mean(d[(d < width)] ** 2))


def gradient_minima(
    r: np.ndarray,
    lnv2: np.ndarray,
    mask_vr: np.ndarray,
    n_points: int,
    r_min: float,
    r_max: float,
) -> Tuple[np.ndarray]:
    """Computes the r-lnv2 gradient and finds the minimum as a function of `r`
    within the interval `[r_min, r_max]`

    Parameters
    ----------
    r : np.ndarray
        Radial separation
    lnv2 : np.ndarray
        Log-kinetic energy
    mask_vr : np.ndarray
        Mask for the selection of radial velocity
    n_points : int
        Number of minima points to compute
    r_min : float
        Minimum radial distance
    r_max : float
        Maximum radial distance

    Returns
    -------
    Tuple[np.ndarray]
        Radial and minima coordinates.
    """
    r_edges_grad = np.linspace(r_min, r_max, n_points + 1)
    grad_r = 0.5 * (r_edges_grad[:-1] + r_edges_grad[1:])
    grad_min = np.zeros(n_points)
    for i in range(n_points):
        r_mask = (r > r_edges_grad[i]) * (r < r_edges_grad[i + 1])
        hist_yv, hist_edges = np.histogram(lnv2[mask_vr * r_mask], bins=200)
        hist_lnv2 = 0.5 * (hist_edges[:-1] + hist_edges[1:])
        hist_lnv2_grad = np.gradient(hist_yv, np.mean(np.diff(hist_edges)))
        lnv2_mask = (1.0 < hist_lnv2) * (hist_lnv2 < 2.0)
        grad_min[i] = hist_lnv2[lnv2_mask][np.argmin(hist_lnv2_grad[lnv2_mask])]

    return grad_r, grad_min

@timer
def calibrate_finder(
    n_seeds: int,
    r_max: float,
    boxsize: float,
    subsize: float,
    file_seeds: str,
    path: str,
    part_mass: float,
    rhom: float,
    n_points: int = 20,
    perc: float = 0.98,
    width: float = 0.05,
    grad_lims: tuple = (0.2, 0.5),
):
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
        Mass density of the universe
    n_points : int, optional
        Number of minima points to compute, by default 20
    perc : float, optional
        Target percentile for the positive radial velocity calibration, 
        by default 0.98
    width : float, optional
        Band width for the negattive radial velocity calibration, 
        by default 0.05
    """
    r, vr, lnv2 = get_calibration_data(
        n_seeds=n_seeds,
        r_max=r_max,
        boxsize=boxsize,
        subsize=subsize,
        file_seeds=file_seeds,
        path=path,
        part_mass=part_mass,
        rhom=rhom
    )

    mask_vr_neg = (vr < 0)
    mask_vr_pos = ~mask_vr_neg
    mask_r = r < 2.0

    # For vr > 0 ===============================================================
    r_grad, min_grad = gradient_minima(r, lnv2, mask_vr_pos, n_points, *grad_lims)
    # Find slope by fitting to the minima.
    popt, _ = curve_fit(lambda x, m, b: m * x + b, r_grad, min_grad, p0=[-1, 2])
    m_pos, b01 = popt

    # Find intercept by finding the value that contains 96% of particles below
    # the line.
    res = minimize(
        cost_percentile,
        1.1 * b01,
        bounds=((0.8 * b01, 3.0),),
        args=(r[mask_vr_pos * mask_r], lnv2[mask_vr_pos * mask_r], m_pos, perc),
        method='Nelder-Mead',
    )
    b_pos = res.x[0]

    # For vr < 0 ===============================================================
    r_grad, min_grad = gradient_minima(r, lnv2, mask_vr_neg, n_points, *grad_lims)
    # Find slope by fitting to the minima.
    popt, _ = curve_fit(lambda x, m, b: m * x + b, r_grad, min_grad, p0=[-1, 2])
    m_neg, b02 = popt

    # Find intercept by finding the value that maximizes the perpendicular
    # distance between points to the line.
    res = minimize(
        cost_perp_distance,
        0.75 * b02,
        bounds=((1.2, b02),),
        args=(r[mask_vr_neg], lnv2[mask_vr_neg], m_neg, width),
        method='Nelder-Mead',
    )
    b_neg = res.x[0]

    with h5.File(path + 'calibration_pars.hdf5', 'w') as hdf:
        hdf.create_dataset('pos', data=[m_pos, b_pos])
        hdf.create_dataset('neg', data=[m_neg, b_neg])

    return


if __name__ == "__main__":
    pass
