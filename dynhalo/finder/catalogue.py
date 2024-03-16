from typing import Tuple, List, Union

import numpy as np

from dynhalo.utils import G_gravity


def find_r200_m200(
    r_rel: np.ndarray,
    part_mass: float,
    rhom: float,
) -> Tuple[float]:
    """Find R200 and M200 around a given seed using all particles.

    Parameters
    ----------
    r_rel : np.ndarray
        Relative coordinates
    part_mass : float
        Mass per particle
    rhom : float
        Matter density

    Returns
    -------
    Tuple[float]
        R200 and M200
    """
    dists = np.sqrt(np.sum(np.square(r_rel), axis=1))
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
) -> np.ndarray[bool]:
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
    np.ndarray[bool]
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




if __name__ == "__main__":
    pass
