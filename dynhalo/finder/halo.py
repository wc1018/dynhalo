from typing import Tuple

import numpy as np


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
