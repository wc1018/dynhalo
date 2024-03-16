from typing import Tuple

import numpy as np


def relative_coordinates(
    x0: np.ndarray,
    x: np.ndarray,
    boxsize: float,
    periodic: bool = True
) -> float:
    """Returns the coordinates x relative to x0 accounting for periodic boundary
    conditions

    Parameters
    ----------
    x0 : np.ndarray
        Reference position in cartesian coordinates
    x : np.ndarray
        Position array (N, 3)
    boxsize : float
        Size of simulation box
    periodic : bool, optional
        Set to True if the simulation box is periodic, by default True

    Returns
    -------
    float
        Relative positions
    """
    if periodic:
        return (x - x0 + 0.5*boxsize) % boxsize - 0.5*boxsize
    return x - x0


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


if __name__ == "__main__":
    pass
