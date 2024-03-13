from itertools import repeat
from multiprocessing.pool import Pool
from typing import Callable, Tuple
from warnings import filterwarnings

import numpy as np
from nbodykit.lab import cosmology
from scipy.interpolate import interp1d
from velocileptors.LPT.cleft_fftw import CLEFT
from velocileptors.Utils.qfuncfft import loginterp
from ZeNBu.Utils.spherical_bessel_transform_fftw import SphericalBesselTransform

from dynhalo.corrfunc.model import error_func_pos_incr

filterwarnings("ignore")


def zeldovich_approx_corr_func_prediction(
    h: float, Om: float, Omb: float, ns: float, sigma8: float, z: float = 0,
    power_spectra: bool = True
) -> Tuple[Callable]:
    """Returns the linear and Zel'dovich approximation power spectra and
    correlation functions.

    Parameters
    ----------
    h : float
        H0 / 100, where H0 is the Hubble parameter.
    Om : float
        Matter density
    Omb : float
        Baryonic matter density
    ns : float
        Spectral index
    sigma8 : float
        _description_
    z : float, optional
        Cosmological Redshift, by default 0
    power_spectra : bool
        If True, returns the linear and ZA power spectra

    Returns
    -------
    Tuple[Callable]
        Power spectra and correlation functions as callables to evaluate on
        arbitrary k and r grids.
    """
    c = cosmology.Cosmology(
        h=h,
        Omega0_b=Omb,
        Omega0_cdm=Om - Omb,
        n_s=ns,
        k_max_tau0_over_L_max=15.0,
    ).match(sigma8=sigma8)

    # Linear and ZA power spectra
    pk_lin = cosmology.LinearPower(c, redshift=z, transfer="CLASS")
    pk_zel = cosmology.ZeldovichPower(c, redshift=z)

    # Correlation functions
    cf_lin = cosmology.CorrelationFunction(pk_lin)
    cf_zel = cosmology.CorrelationFunction(pk_zel)

    if power_spectra:
        return cf_lin, cf_zel, pk_lin, pk_zel
    else:
        return cf_lin, cf_zel


def eft_tranform(k: np.ndarray) -> SphericalBesselTransform:
    """Creates a spherical Bessel transform object to compute fast Fourier 
    transformations.

    Parameters
    ----------
    k : np.ndarray
        Fourier modes.

    Returns
    -------
    SphericalBesselTransform

    """
    return SphericalBesselTransform(k, L=5, low_ring=True, fourier=True)


def eft_counter_term_power_spec_prediction(
    k_lin: np.ndarray,
    p_lin: np.ndarray,
    cs: float = 0,
    k_min: float = 0,
) -> Tuple[np.ndarray]:
    """Computes the 1-Loop LPT power spectrum prediction from linear power
      spectrum up to the counter term.

    Parameters
    ----------
    k_lin : np.ndarray
        Fourier modes.
    p_lin : np.ndarray
        Linear power spectrum evaluated at each k
    cs : float, optional
        Speed sound for the counter term, by default 0
    k_min : float
        Smallest k-mode with power. All scales below `k_min` are set to have 
        zero power, by default 0

    Returns
    -------
    Tuple[np.ndarray]
        Correlation function prediction and r
    """
    cleft = CLEFT(k_lin, p_lin)
    cleft.make_ptable(nk=400)

    # 1-loop matter power spectrum
    lptpk = cleft.pktable[:, 1]
    # Counter term contribution
    cterm = cleft.pktable[:, -1]
    # Model evaluation k modes
    kcleft = cleft.pktable[:, 0]

    # Add counter term
    if cs != 0:
        k_factor = kcleft**2 / (1 + kcleft**2)
        lptpk += cs * k_factor * cterm

    eft_pred = loginterp(kcleft, lptpk)(k_lin)
    # The EFT prediction is noisy at low k due to the attenuation, so we force
    # large-scale power to be zero
    eft_pred[k_lin < k_min] = 0

    return eft_pred


def eft_counter_term_corr_func_prediction(
    k_lin: np.ndarray,
    p_lin: np.ndarray,
    cs: float = 0,
    k_min: float = 0,
) -> Tuple[np.ndarray]:
    """Computes the 1-Loop LPT correlation function prediction from linear power
      spectrum up to the counter term.

    Parameters
    ----------
    k_lin : np.ndarray
        Fourier modes.
    p_lin : np.ndarray
        Linear power spectrum evaluated at each k
    cs : float, optional
        Speed sound for the counter term, by default 0
    k_min : float
        Smallest k-mode with power. All scales below `k_min` are set to have 
        zero power, by default 0

    Returns
    -------
    Tuple[np.ndarray]
        Correlation function prediction and r
    """
    # Get power spectrum
    eftpred = eft_counter_term_power_spec_prediction(k_lin, p_lin, cs=cs, k_min=k_min)
    # Hankel transform object
    sph = eft_tranform(k_lin)
    r_eft, xi_eft = sph.sph(0, eftpred)

    return r_eft, xi_eft[0]


def power_spec_box_effect(
    k: np.ndarray,
    pk: np.ndarray,
    boxsize: float,
    lamb: float
) -> np.ndarray:
    """Truncates the power spectrum at small k accounting for the effects of a 
    finite simulation box.

    Parameters
    ----------
    k : np.ndarray
        Fourier modes
    pk : np.ndarray
        Power spectrum evaluated at each k
    boxsize : float
        Simulation box size
    lamb : float
        Attenuation factor at small k

    Returns
    -------
    np.ndarray
        Attenuated power spectrum such that $P(k \rightarrow 0) = 0$
    """
    rbox = boxsize * np.cbrt(3. / 4. / np.pi)
    phat = (1 - np.exp(-lamb * (rbox * k) ** 2)) * pk
    return phat


def heaviside_step(
    lamb: float,
    theta: float,
    phi: float, 
    boxsize: float,
) -> bool:
    """Computes the Heaviside step function in spherical coordinates

    Parameters
    ----------
    lamb : float
        Wave length in unites of `boxsize`
    theta : float
        Azimuthal angle
    phi : float
        Axial angle
    boxsize : float
        Simulation box size

    Returns
    -------
    bool
        True if the wave length fits inside the box. False otherwise
    """
    y1 = np.abs(lamb * np.sin(theta) * np.cos(phi)) < boxsize
    y2 = np.abs(lamb * np.sin(theta) * np.sin(phi)) < boxsize
    y3 = np.abs(lamb * np.cos(theta)) < boxsize
    return y1 & y2 & y3


def k_modes_fraction_monte_carlo(
    lamb: float,
    theta: np.ndarray,
    phi: np.ndarray, 
    boxsize: float,
) -> float:
    """Returns the Monte-Carlo integral of the fraction of wave lengths that fit 
    inside the box across all possible angles

    The integral evaluated is 

    \begin{align*}
    f(\lambda) &=  \frac{1}{4\pi} \int d\Omega\, \Theta(\lambda, \theta, \phi) \\
    & = \int d\Omega\, \Theta(\lambda, \theta, \phi) P(\theta, \phi)
    \end{align*}

    where $P(\theta, \phi)=\frac{1}{4\pi}$ is a uniform distribution over the sphere.

    The Monte Carlo integral for this problem looks as follows:

    \begin{equation*}
    f(\lambda) =  \frac{1}{N} \sum_{\theta, \phi} \Theta(\lambda, \theta, \phi)
    \end{equation*}

    where the angles are uniformly distributed over the sphere. That is:
    \begin{align*}
    \cos\theta &= U(0, 1) \\
    \phi & = U(0, 2\pi)
    \end{align*}

    Parameters
    ----------
    lamb : float
        Wave length in unites of `boxsize`
    theta : float
        Randomly sampled azimuthal angles
    phi : float
        Randomly samlped axial angles
    boxsize : float
        Simulation box size

    Returns
    -------
    float
        Fraction of wave lengths with length `lamb` that fit inside the box.
    """
    return np.mean(heaviside_step(lamb, theta, phi, boxsize))


def power_spec_box_effect_k_modes(
    k: np.ndarray,
    pk: np.ndarray,
    boxsize: float,
    n_draws: int = 1_000_000,
    n_modes: int = 500
) -> np.ndarray:
    """Returns the power spectrum `pk` such that large scales are attenuated by 
    the fraction of Fourier k modes that fit inside the box. This is done 
    through integrating over all angles for each k within the size of the box 
    and the main diagonal of the box.

    Parameters
    ----------
    k : np.ndarray
        Fourier modes
    pk : np.ndarray
        Power spectrum evaluated at each k
    boxsize : float
        Simulation box size
    n_draws : int, optional
        Number of random draws for each angle, by default 1_000_000
    n_modes : int, optional
        Number of subdivisions between the mode corresponding to the size of the
        box and the main diagonal mode, by default 500

    Returns
    -------
    np.ndarray
        Attenuated power spectrum
    """
    # Maximum wave mode that fits in the main diagonal of the box
    lamb_max = np.sqrt(3.) * boxsize

    # Draw random samples frtom
    thetas = np.arccos(np.random.uniform(size=n_draws))
    phis = np.random.uniform(high=2.*np.pi, size=n_draws)

    # Create a linear grid of wave modes
    lambda_modes = np.linspace(0.95*boxsize, 1.05*lamb_max, n_modes)
    # The corresponding Fourier modes
    k_modes = 2.*np.pi / lambda_modes

    # Compute the Monte-Carlo integral for the fraction of modes that fit in the
    # box given the wave length
    # args = (thetas, phis, boxsize)
    with Pool(16) as pool:
        frac = pool.starmap(k_modes_fraction_monte_carlo,
                            zip(
                                lambda_modes,
                                repeat(thetas, n_modes),
                                repeat(phis, n_modes),
                                repeat(boxsize, n_modes)
                            ))
    frac = np.array(frac)

    # Interpolate the fraction as a function of Fourier modes and evaluate the
    # fraction in the given k modes
    frac_phat = interp1d(k_modes, frac, fill_value='extrapolate')(k)

    return frac_phat * pk


def loglike_cs(cs: float, data: Tuple[float]) -> float:
    """Log-likelihood for the sound speed parameter of the counter term.

    Parameters
    ----------
    cs : float
        Sound speed
    data : Tuple[float]
        (k, pk, r, xi, cov)

    Returns
    -------
    float

    """
    # Check parameter prior
    if cs < 0:
        return -np.inf

    # Unpack data
    k, pk, r, xi, cov = data
    xi_pred = interp1d(*eft_counter_term_corr_func_prediction(k, pk, cs=cs))
    # Compute chi2
    d = xi - xi_pred(r)
    return -np.dot(d, d / np.diag(cov))


def loglike_B(B: float, data: Tuple[np.ndarray]) -> float:
    """Log-likelihood for the ratio parameter between ZA and data.

    Parameters
    ----------
    B : float
        Ratio parameter
    data : Tuple[float]
        (xi, xi_pred)

    Returns
    -------
    float

    """
    # Check parameter prior
    if B < 0:
        return -np.inf

    # Unpack data
    xi, xi_pred = data
    # Compute chi2
    d = xi - B * xi_pred
    return -np.dot(d, d)


def find_cs(
    k_lin: np.ndarray, 
    p_lin: np.ndarray, 
    r: np.ndarray, 
    xi: np.ndarray, 
    xi_cov: np.ndarray, 
    r_min: float = 40, 
    r_max: float = 80
) -> float:
    """Finds the optimal cs value for the given power spectrum

    Parameters
    ----------
    k_lin : np.ndarray
        Fourier modes.
    p_lin : np.ndarray
        Linear power spectrum evaluated at each k
    r : np.ndarray
        Radial points where the correlation function was measured
    xi : np.ndarray
        Measured correlation function
    xi_cov : np.ndarray
        Covariance matrix of the measured correlation function
    r_min : float, optional
        Lower bound for the fit, by default 40
    r_max : float, optional
        Upper bound for the fit, by default 80

    Returns
    -------
    float
        Speed sound for the counter term
    """
    r_mask = (r_min < r) & (r_max < 80)
    args = (k_lin, p_lin, r[r_mask], xi[r_mask], xi_cov[r_mask, :][:, r_mask])
    # Define grids to estimate cs
    ngrid = 16
    grid = np.logspace(0, 1.2, ngrid)
    with Pool(16) as p:
        loglike_grid = p.starmap(loglike_cs, zip(grid, repeat(args, ngrid)))
    cs_max = grid[np.argmax(loglike_grid)]
    # Refine grid around cs_max with 10% deviation
    ngrid = 80
    grid = np.linspace(0.9*cs_max, 1.1*cs_max, ngrid)
    with Pool(16) as p:
        loglike_grid = p.starmap(loglike_cs, zip(grid, repeat(args, ngrid)))
    cs_max = grid[np.argmax(loglike_grid)]
    return cs_max


def find_B(
    xi_data: np.ndarray,
    xi_zel: np.ndarray
) -> float:
    """Find the optimal value for the offset B

    Parameters
    ----------
    xi_data : np.ndarray
        Measured correlation function
    xi_zel : np.ndarray
        Zel'dovich approximaiton prediction

    Returns
    -------
    float
        Offset parameter
    """
    B_grid = np.linspace(0.8, 1.2, 10_000)
    loglike_grid = [loglike_B(b, (xi_data, xi_zel))
                    for b in B_grid]
    B_max = B_grid[np.argmax(loglike_grid)]
    return B_max


def xi_large_construct(
    r: np.ndarray,
    xi_zel: np.ndarray,
    xi_eft: np.ndarray,
    b: float,
    fmu: float = 40.,
    fsig: float = 3.,
) -> np.ndarray:
    """Constructs the function \xi_large for given LTP and Zel'dovich 
    Approximation models, as a smooth transition between both terms.

    xi_large = (1-f)\xi_zel + f\xi_LTP

    where f is an error function.

    Parameters
    ----------
    r : np.ndarray
        Radial points of the measured correlation function
    xi_zel : np.ndarray
        Zel'dovich approximation prediction at each r
    xi_eft : np.ndarray
        LTP prediction at each r
    b : float
        Offset parameter
    fmu : float, optional
        Error function mean/offset parameter, by default 40.
    fsig : float, optional
        Error function width parameter, by default 3.

    Returns
    -------
    np.ndarray
        Correlation function for all scales
    """
    erf_transition = error_func_pos_incr(r, 1.0, fmu, fsig)
    xi_large = (1.0 - erf_transition) * b * xi_zel + \
        erf_transition * xi_eft
    return xi_large


def xi_large_estimation_from_data(
    r: np.ndarray,
    xi: np.ndarray,
    xi_cov: np.ndarray,
    h: float,
    Om: float,
    Omb: float,
    ns: float,
    sigma8: float,
    boxsize: float,
    z: float = 0,
    large_only: bool = True,
    power_spectra: bool = False,
) -> Tuple[np.ndarray]:
    """Computes the large scale limit correlation function using 1-Loop LPT.

    Parameters
    ----------
    r : np.ndarray
        Radial points of the measured correlation function
    xi : np.ndarray
        Measured correlation function from a simulation box
    xi_cov : np.ndarray
        Covariance matrix of the measured correlation function
    h : float
        H0 / 100, where H0 is the Hubble parameter.
    Om : float
        Matter density
    Omb : float
        Baryonic matter density
    ns : float
        Spectral index
    sigma8 : float
        _description_
    boxsize : float
        Length of one side of the simulation box.
    z : float, optional
        Cosmological Redshift, by default 0
    large_only : bool, optional
        if True, only returns `xi_large`, by default True
    power_spectra : bool, optional
        If True, also returns the power spectra, by default False

    Returns
    -------
    Tuple[np.ndarray]

    """
    # Compute ZA
    xi_lin_call, xi_zel_call, pk_lin_call, pk_zel_call = (
        zeldovich_approx_corr_func_prediction(
            h=h, Om=Om, Omb=Omb, ns=ns, sigma8=sigma8, z=z
        )
    )
    k_lin = np.logspace(-4, 3, 1000, base=10)
    p_lin = pk_lin_call(k_lin)

    # Find the best value of cs that minimizes the eft prediction error at
    # intermediate scales
    cs_max = find_cs(k_lin, p_lin, r, xi, xi_cov)

    # Account for the box size effects on large scale Fourier modes
    p_hat = power_spec_box_effect_k_modes(k_lin, p_lin, boxsize)

    # Compute the 1-loop EFT approx.
    k_min = 2 * np.pi / np.sqrt(3) / boxsize
    p_eft = eft_counter_term_power_spec_prediction(k_lin, p_hat, cs=cs_max, k_min=k_min)
    r_eft, xi_eft = eft_counter_term_corr_func_prediction(
        k_lin, p_hat, cs=cs_max, k_min=k_min)

    # Evaluate ZA in the same grid as EFT
    p_zel = pk_zel_call(k_lin)
    xi_lin = xi_lin_call(r_eft)
    xi_zel = xi_zel_call(r_eft)

    # Find the ratio between EFT and ZA
    r_mask = (30 < r) & (r < 50)
    B_max = find_B(xi[r_mask], interp1d(r_eft, xi_zel)(r[r_mask]))

    # Construct xi large
    xi_large = xi_large_construct(r_eft, xi_zel, xi_eft, B_max)

    power_spectra = (k_lin, p_lin, p_hat, p_eft, p_zel)
    corr_func = (r_eft, xi_lin, xi_eft, xi_zel,
                 #  xi_large, B_max, cs_max, lamb_max)
                 xi_large, B_max, cs_max)
    # Return all quantities
    if large_only:
        return r_eft, xi_large
    else:
        if power_spectra:
            return power_spectra, corr_func
        else:
            return corr_func


if __name__ == "__main__":
    pass
