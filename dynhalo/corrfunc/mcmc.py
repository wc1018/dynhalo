import os
from multiprocessing.pool import Pool
from typing import Callable, List, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
from chainconsumer import ChainConsumer
from emcee.autocorr import integrated_time
from emcee.backends import HDFBackend
from emcee.ensemble import EnsembleSampler


def walker_init(
    pars_init: Union[np.ndarray, List, Tuple],
    log_prob: Callable,
    log_prob_args: Union[List, Tuple],
    n_walkers: int,
) -> np.ndarray:
    """Checks initial point for the chain and that all walkers are
    initialized correctly around this point.

    Parameters
    ----------
    pars_init : Union[np.ndarray, List, Tuple]
        Initial point for each parameter. The number of dimensions is inferred 
        from its length 
    log_prob : Callable
        Log probability function
    log_prob_args : Union[List, Tuple]
        Arguments passed to the `log_prob` function
    n_walkers : int
        Number of walkers per dimension

    Returns
    -------
    np.ndarray
        The initial position for all the walkers. The array has shape 
        (n_walkers, n_dim)

    Raises
    ------
    ValueError
        If log-posterior returned infinity at inital point.
    ValueError
        If log-posterior returned infinity for any of the walkers.
    """
    # Get the number of dimensions (parameters)
    n_dim = len(pars_init)

    # Check initial point.
    lnpost_init = log_prob(pars_init, *log_prob_args)

    if not np.isfinite(lnpost_init):
        raise ValueError("Initial point returned infinity")
    else:
        print(f"\t Initial log posterior {lnpost_init:.2f}")

    # Initialize walkers around initial point with a 10% uniform scatter.
    rand = np.random.uniform(low=-1, size=(n_walkers, n_dim))
    walkers_init = pars_init * (1.0 + 0.1 * rand)

    # Check walkers.
    lnlike_inits = [log_prob(walkers_init[i], *log_prob_args)
                    for i in range(n_walkers)]

    if not all(np.isfinite(lnlike_inits)):
        raise ValueError("Some walkers are not properly initialized.")

    return walkers_init


def walker_reinit(
    chain: np.ndarray,
    log_prob: np.ndarray
) -> np.ndarray:
    """Apply a median absolute deviation (MAD) criteria to select 'good' and
    'bad' walkers from a chain, where 'bad' walkers get new positions from a
    previous state of any 'good' walker. Returns a list with the new
    positions for all walkers.

    Parameters
    ----------
    chain : np.ndarray
        MCMC chain of shape (n_steps, n_walkers, n_dim)
    log_prob : np.ndarray
        Log probability of each walker of shape (n_steps, n_walkers)

    Returns
    -------
    np.ndarray
        New position for all walkers. If the walker is deemed 'good' its 
        position is unchanged. 
    """
    nsteps, nwalkers, ndim = chain.shape
    if nsteps >= 50:
        ntail = 50
    else:
        ntail = nsteps

    # Append log-likelihood to the samples in order to compute all medians
    # in a vectorized manner.
    samples = np.zeros((nsteps, nwalkers, ndim + 1))
    samples[:, :, :-1] = chain
    samples[:, :, -1] = log_prob

    # Median for parameter i and walker a (nwalkers x ndim+1)
    theta_ia = np.median(samples, axis=0)
    # Median for parameter i (median over all walkers) (ndim+1)
    theta_i = np.median(theta_ia, axis=0)
    # Median absolute deviation (MAD)
    dtheta_ia = np.abs(theta_ia - theta_i)
    sigma = 1.4826 * np.median(dtheta_ia, axis=0)

    # Select 'good' walkers.
    good_walkers = np.zeros(nwalkers, dtype=bool)
    for i in range(nwalkers):
        # If inequality is satisfied for any parameter (or log-likelihood),
        # label the walker as 'good'.
        good_walkers[i] = all(dtheta_ia[i, :] / sigma < 3)
    bad_walkers = ~good_walkers
    good_walkers = good_walkers

    # Return the last chain step if there are no 'bad' walkers
    if bad_walkers.sum() == 0:
        return chain[-1, :, :]

    # Draw n_bad_walker samples from 'good' walkers earlier in time and
    # replace the 'bad' walkers final positons per dimension.
    pos_new = chain[-1, :, :]
    for i in range(ndim):
        pos_new[bad_walkers, i] = np.random.choice(
            chain[-ntail:, good_walkers, i].flatten(), size=bad_walkers.sum()
        )

    return pos_new


def run_burn(
    path: str,
    pars_init: Union[np.ndarray, List, Tuple],
    n_burn: int,
    n_burn_steps: int,
    n_walkers: int,
    log_prob: Callable,
    log_prob_args: Union[List, Tuple],
    n_cpu: int = 1,
) -> np.ndarray:
    # Get the number of dimensions (parameters)
    n_dim = len(pars_init)

    # Initialize walkers
    walker_pos = walker_init(pars_init=pars_init, log_prob=log_prob,
                             log_prob_args=log_prob_args, n_walkers=n_walkers)

    # Initialize backend to save chain state
    backend_burn = HDFBackend(path + 'burn.hdf5', name='burn')

    for burn_step in range(n_burn_steps):
        backend_burn.reset(n_walkers, n_dim)
        with Pool(n_cpu) as pool:
            # Instantiate sampler
            sampler = EnsembleSampler(
                nwalkers=n_walkers,
                ndim=n_dim,
                log_prob_fn=log_prob,
                pool=pool,
                backend=backend_burn,
                args=log_prob_args,
            )

            sampler.run_mcmc(
                initial_state=walker_pos,
                nsteps=n_burn,
                progress=True,
                progress_kwargs={
                    'desc': f"Burn {burn_step}",
                    'ncols': 100,
                    'colour': 'yellow'
                }
            )

        # Re-initialize the walkers by sampling from the last 20% of the
        # burnin steps. Discard first 80% of points
        burn_tail = int(0.8 * n_burn)
        walker_pos = walker_reinit(
            sampler.get_chain(discard=burn_tail),
            sampler.get_log_prob(discard=burn_tail),
        )

    return


def run_chain(
    path: str,
    n_steps: int,
    n_walkers: int,
    n_dim: int,
    log_prob: Callable,
    log_prob_args: Union[List, Tuple],
    n_cpu: int = 1,
    name: str = None
) -> None:
    # Initialize walkers to the last step if the burn chain.
    with h5py.File(path + 'burn.hdf5', 'r') as hdf:
        walker_pos = hdf['burn/chain'][-1]

    # Initialize backend to save chain state
    backend_burn = HDFBackend(path + 'chain.hdf5', name=name)

    backend_burn.reset(n_walkers, n_dim)
    with Pool(n_cpu) as pool:
        # Instantiate sampler
        sampler = EnsembleSampler(
            nwalkers=n_walkers,
            ndim=n_dim,
            log_prob_fn=log_prob,
            pool=pool,
            backend=backend_burn,
            args=log_prob_args
        )

        sampler.run_mcmc(
            initial_state=walker_pos,
            nsteps=n_steps,
            progress=True,
            progress_kwargs={
                'desc': f"Chain {name}",
                'ncols': 100,
                'colour': 'green'
            }
        )

    # Delete burn file.
    os.remove(path + 'burn.hdf5')

    return


def summary(
    path: str,
    name: str,
    n_dim: int,
    mle_name: str,
    mle_file: str = 'mle.hdf5',
    p_labs: list = None,
    plot_path: str = None,
    plot_name: str = None,
) -> None:
    if plot_path is None:
        plot_path = path

    sampler = HDFBackend(path + f'chain.hdf5', name=name, read_only=True)
    flat_samples = sampler.get_chain(flat=True)
    log_prob = sampler.get_log_prob()

    tau = [
        integrated_time(flat_samples[:, i], c=150, tol=50)[0]
        for i in range(n_dim)
    ]
    print(f"Autocorrelation length {max(tau):.2f}")

    # Setup chainconsumer for computing MLE parameters
    c = ChainConsumer()
    if p_labs:
        c.add_chain(flat_samples, posterior=log_prob.reshape(-1),
                    parameters=p_labs)
    else:
        c.add_chain(flat_samples, posterior=log_prob.reshape(-1))
    c.configure(
        summary=False,
        sigmas=[1, 2],
        colors='k',
        # tick_font_size=10,
        # max_ticks=3,
        usetex=True,
    )

    # Compute max posterior and quantiles (16, 50, 84).
    quantiles = np.zeros((n_dim, 3))
    max_posterior = np.zeros(n_dim)
    for i, ((_, v1), (_, v2)) in enumerate(
        zip(
            c.analysis.get_summary().items(),
            c.analysis.get_max_posteriors().items(),
        )
    ):
        quantiles[i, :] = v1
        max_posterior[i] = v2
    cov = c.analysis.get_covariance()[-1]
    corr = c.analysis.get_correlations()[-1]

    # Save to file.
    with h5py.File(path + mle_file, 'a') as hdf:
        for ds, var in zip(
            ['quantiles/', 'max_posterior/', 'covariance/', 'correlations/'],
            [quantiles, max_posterior, cov, corr],
        ):
            name_ = ds + mle_name
            # Overwirte existing values.
            if name_ in hdf.keys():
                sp = hdf[name_]
                sp[...] = var
            # Create dataset otherwise.
            else:
                hdf.create_dataset(name_, data=var)

    return None


if __name__ == '__main__':
    pass
