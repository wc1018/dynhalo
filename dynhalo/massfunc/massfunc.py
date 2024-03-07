import math
from typing import List, Tuple, Union

import numpy as np
from colossus.cosmology import cosmology as ccosmo
from colossus.lss import peaks as cpeaks

from scipy.interpolate import interp1d
from scipy.integrate import simpson
from dynhalo.corrfunc.bins import partition_box


def set_colossus_cosmology(params) -> ccosmo.Cosmology:
    ccosmo.addCosmology(cosmo_name='mycosmo', params=params)
    cosmo = ccosmo.setCosmology(cosmo_name='mycosmo')
    return cosmo


def h_parameter(z: float, cosmo: ccosmo.Cosmology) -> float:
    return cosmo.Hz(z=z) / 100.


def peak_height(mass: float, z: float) -> float:
    return cpeaks.peakHeight(M=mass, z=z)


def measure_mass_function_jk(
    positions: np.ndarray,
    masses: np.ndarray,
    boxsize: float,
    gridsize: float,
    nu_edges: Union[List, Tuple, np.ndarray],
    z: float,
) -> np.ndarray:
    # Number of grid cells per side
    cells_per_side = int(math.ceil(boxsize / gridsize))
    # Number of jackknife samples. One sample per cell
    n_jk_samples = cells_per_side**3
    # Number of radial bins
    n_bins = nu_edges.shape[0] - 1

    # Partition box
    data_cell_id = partition_box(
        data=positions, boxsize=boxsize, gridsize=gridsize)

    f_nu_samples = np.zeros((n_jk_samples, n_bins))
    for sample in range(n_jk_samples):
        nu = peak_height(mass=masses[data_cell_id[sample]], z=z)
        f_nu_samples[sample], _ = np.histogram(a=nu, bins=nu_edges)

    # Compute mean from all jk samples
    f_nu_mean = np.mean(f_nu_samples, axis=0)

    # Compute covariance matrix of the radial bins using all jk samples
    f_nu_cov = (float(n_jk_samples) - 1.0) * np.cov(f_nu_samples.T, bias=True)

    # Compute the total using all items
    f_nu, _ = np.histogram(a=peak_height(mass=masses, z=z), bins=nu_edges)

    return f_nu, f_nu_samples, f_nu_mean, f_nu_cov


def moving_barrier_model(
    nu_edges: float,
    delta_c: float,
    a: float,
    b: float,
    rhom: float,
    mass_from_nu: interp1d,
    boxsize: float,
) -> float:
    delta_sc = 1.686
    c = delta_c / delta_sc
    n_points = len(nu_edges) - 1
    eval_at_nu = np.zeros(n_points)
    for i in range(n_points):
        nu_temp = np.linspace(nu_edges[i], nu_edges[i + 1], 100)
        nu_tilde = nu_temp * (c + b * np.exp(-nu_temp / a) / delta_sc)
        integrand = np.sqrt(2./np.pi) * np.exp(-0.5 * nu_tilde**2) * \
            rhom / mass_from_nu(nu_temp) * boxsize**3 * \
            (c + b / delta_sc * np.exp(-nu_temp / a) * (1. - nu_temp / a))
        eval_at_nu[i] = simpson(integrand, nu_temp)

    return eval_at_nu


def moving_barrier_lnlike(pars, *data):
    # Check parameter priors
    delta_c, a, b, logd = pars
    out_of_prior_neg = any([p < 0 for p in (delta_c, a, b)])
    out_of_prior_lrg = any([a > 10, b > 100])
    out_of_prior_logd = (-4 > logd) | (logd > 0)
    if out_of_prior_neg or out_of_prior_lrg or out_of_prior_logd:
        return -np.inf

    # Unpack data and evaluate parameters
    x_edges, y, covy, rhom, mass_from_nu, boxsize = data
    delta = np.power(10., logd)

    y_model = moving_barrier_model(
        x_edges, delta_c, a, b, rhom, mass_from_nu, boxsize)
    cov = covy + np.diag((delta*y)**2)

    mask = np.diag(covy) != 0
    y = y[mask]
    y_model = y_model[mask]
    cov = cov[mask, :][:, mask]

    d = y - y_model
    log_det_cov = np.log(np.linalg.det(cov))
    chi2 = np.dot(d, np.linalg.solve(cov, d))

    return -chi2 - log_det_cov


if __name__ == '__main__':
    pass
