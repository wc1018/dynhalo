import math
from typing import List, Tuple, Union
from warnings import filterwarnings

import numpy as np
from Corrfunc.theory import DD as countDD
from tqdm import tqdm

filterwarnings('ignore')

__all__ = ["generate_bins", "generate_bin_str",
           "partition_box", "process_DD_pairs"]

def generate_bins(
    bmin: float,
    bmax: float,
    nbins: int,
    logspaced: bool = True,
    soft: float = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates equally spaced bins in linear or logarithmic (base 10) space.

    Parameters
    ----------
    bmin : float
        Minimum or lower bound (inclusive)
    bmax : float
        Maximum or upper bound (inclusive)
    nbins : int
        Number of bins to generate
    logspaced : bool, optional
        Generate log-spaced bins, by default True
    soft : float, optional
        If there is softening value, then bmin >= soft, by default 0

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Bins and edges in units of bmin and bmax
    """
    if bmin < soft:
        bmin = soft
    # Equally log-spaced bins
    if bmin == 0:
        logspaced = False

    if logspaced:
        r_edges = np.logspace(
            start=np.log10(bmin),
            stop=np.log10(bmax),
            num=nbins + 1,
            base=10,
        )
    else:
        r_edges = np.linspace(
            start=bmin,
            stop=bmax,
            num=nbins + 1,
        )
    # Bin middle point
    rbins = 0.5 * (r_edges[1:] + r_edges[:-1])
    return rbins, r_edges


def generate_bin_str(bin_edges: Union[List[float], Tuple[float]]) -> str:
    """Generates a formatted string 'min-max' from bin edges. Intended to be 
    compatible with HDF dataset naming and LaTeX format.

    Parameters
    ----------
    bin_edges : Union[List[float], Tuple[float]]
        A 2-tuple or list with minimum or lower bound and maximum or upper bound
        of the bin

    Returns
    -------
    str
        _description_

    Raises
    ------
    ValueError
        If bin_edges is not a list or tuple of length 2.
    """
    if type(bin_edges) in [list, tuple] and len(bin_edges) == 2:
        return f'{bin_edges[0]:.2f}-{bin_edges[-1]:.2f}'
    else:
        raise ValueError("bin_edges must be a list of floats and len=2: [min, max]")


def partition_box(data: np.ndarray, boxsize: float, gridsize: float) -> List[float]:
    """Sorts all data points into a 3D grid with `cells per side = boxsize / gridsize`

    Parameters
    ----------
    data : np.ndarray
        `(N, d)` array with all data points' coordinates, where `N` is the 
        number of data points and `d` the dimensions
    boxsize : float
        Simulation box size (per side)
    gridsize : float
        Grid size (per side)

    Returns
    -------
    List[float]
        
    """
    # Number of grid cells per side.
    cells_per_side = int(math.ceil(boxsize / gridsize))
    # Grid ID for each data point.
    grid_id = (data / gridsize).astype(int, copy=False)
    # Correction for points on the edges.
    grid_id[np.where(grid_id == cells_per_side)] = cells_per_side - 1

    # This list stores all of the particles original IDs in a convenient 3D
    # list. It is kind of a pointer of size n_cpd**3
    data_cell_id = [[] for _ in range(cells_per_side**3)]
    cells = cells_per_side**2 * grid_id[:, 0] + cells_per_side * grid_id[:, 1] + grid_id[:, 2]
    for cell in tqdm(range(np.size(data, 0)), desc="Partitioning box", ncols=100, colour='blue'):
        data_cell_id[cells[cell]].append(cell)

    return data_cell_id


def process_DD_pairs(
    data_1: np.ndarray,
    data_2: np.ndarray,
    data_1_id: list,
    radial_edges: np.ndarray,
    boxsize: float,
    gridsize: float,
    weights_1=None,
    weights_2=None,
    nthreads: int = 4,
) -> np.ndarray:
    """Counts data-data pais by using pre-partitiones box into a 3D grid. 
    Corrfunc does the heavy lifting. Doc is taken from Corrfunc.theory.DD

    Parameters
    ----------
    data_1 : np.ndarray
        The array of X/Y/Z positions for the first set of points. Calculations 
        are done in the precision of the supplied arrays.
    data_2 : np.ndarray
        The array of X/Y/Z positions for the second set of points. Calculations 
        are done in the precision of the supplied arrays.
    data_1_id : list
        Box partitioning 3D grid.
    radial_edges : np.ndarray
        The bins need to be contiguous and sorted in increasing order (smallest
        bins come first).
    boxsize : float
        Size of simulation box
    gridsize : float
        Size of sub-volume or cell of the box
    weights_1 : array_like, real (float/double), optional
        A scalar, or an array of weights of shape `(n_weights, n_positions)` or 
        `(n_positions,)`. If `None` will be set to uniform weights. 
        By default `None`
    weights_2 : array_like, real (float/double), optional
        Same as `weights_1` but for `data_2`, by default None
    nthreads : int, optional
        The number of OpenMP threads to use. Has no effect if OpenMP was not 
        enabled during library compilation, by default 4

    Returns
    -------
    np.ndarray
        _description_
    """
    # Number of grid cells per side
    cells_per_side = int(math.ceil(boxsize / gridsize))
    n_cells = cells_per_side**3
    # Create pairs list for each radial bin for each cell
    dd_pairs = np.zeros((n_cells, radial_edges.size - 1))
    # Fill value if no pairs are found in cell
    zeros = np.zeros(radial_edges.size - 1)

    # Pair counting for all elements in each cell
    for cell in tqdm(range(n_cells), desc="Pair counting", ncols=100, 
                     colour='green'):
        # Get data 1 in cell
        xd1 = data_1[data_1_id[cell], 0]
        yd1 = data_1[data_1_id[cell], 1]
        zd1 = data_1[data_1_id[cell], 2]
        if weights_1 is not None:
            if isinstance(weights_1, np.ndarray):
                wd1 = weights_1[data_1_id[cell]]
            else:
                wd1 = weights_1
        else:
            wd1 = 1.0

        # Get data 2 in cell
        xd2 = data_2[:, 0]
        yd2 = data_2[:, 1]
        zd2 = data_2[:, 2]
        if weights_2 is not None:
            if isinstance(weights_2, np.ndarray):
                wd2 = weights_2[:]
            else:
                wd2 = weights_2
        else:
            wd2 = 1.0

        # Data pair counting if there are elements in cell
        if np.size(xd1) != 0: # and np.size(xd2) != 0:
            autocorr = 0
            # Data pair counting
            DD_counts = countDD(
                autocorr=autocorr,
                nthreads=nthreads,
                binfile=radial_edges,
                X1=xd1,
                Y1=yd1,
                Z1=zd1,
                X2=xd2,
                Y2=yd2,
                Z2=zd2,
                weights1=wd1,
                weights2=wd2,
                weight_type='pair_product',
                periodic=True,
                boxsize=boxsize,
                verbose=False,
            )["npairs"]
            dd_pairs[cell] = DD_counts
        else:
            dd_pairs[cell] = zeros

    return dd_pairs


def tpck_with_jk_from_DD(
    data_1: np.ndarray,
    data_2: np.ndarray,
    data_1_id: list,
    radial_edges: np.ndarray,
    dd_pairs: np.ndarray,
    boxsize: float,
    gridsize: float,
) -> Tuple[np.ndarray]:
    # Set up bins
    nbins = radial_edges.size - 1

    n_cpd = int(math.ceil(boxsize / gridsize))  # Number of cells per dimension
    n_jk = n_cpd**3  # Number of jackknife samples
    d1tot = np.size(data_1, 0)  # Number of objects in d1
    d2tot = np.size(data_2, 0)  # Number of objects in d2
    Vbox = boxsize**3  # Volume of box
    Vshell = np.zeros(nbins)  # Volume of spherical shell
    # Vjk = (N - 1) / N * Vbox
    for m in range(nbins):
        Vshell[m] = (
            4.0 / 3.0 * np.pi *
            (radial_edges[m + 1] ** 3 - radial_edges[m] ** 3)
        )
    n1 = float(d1tot) / Vbox  # Number density of d2
    n2 = float(d2tot) / Vbox  # Number density of d2

    # Some arrays
    ddpairs_removei = np.zeros((n_jk, nbins))
    xi = np.zeros(nbins)
    xi_i = np.zeros((n_jk, nbins))
    meanxi_i = np.zeros(nbins)
    cov = np.zeros((nbins, nbins))

    ddpairs = np.sum(dd_pairs, axis=0)

    ddpairs_removei = ddpairs[None, :] - dd_pairs
    for cell in range(n_jk):
        d1tot_s1 = np.size(data_1, 0) - np.size(data_1[data_1_id[cell]], 0)
        # xi_i[cell] = dd_pairs_i[cell] / (n1 * n2 * Vjk * Vshell) - 1
        xi_i[cell] = ddpairs_removei[cell] / (d1tot_s1 * n2 * Vshell) - 1

    # Compute mean xi from all jk samples
    for i in range(nbins):
        meanxi_i[i] = np.mean(xi_i[:, i])

    # Compute covariance matrix
    cov = (float(n_jk) - 1.0) * np.cov(xi_i.T, bias=True)

    # Compute the total xi
    xi = ddpairs / (n1 * n2 * Vbox * Vshell) - 1

    return xi, xi_i, meanxi_i, cov


def cross_tpcf_jk(
    data_1: np.ndarray,
    data_2: np.ndarray,
    radial_edges: np.ndarray,
    weights_1=None,
    weights_2=None,
    nthreads: int = 16,
    jk_estimates: bool = True,
) -> Tuple[np.ndarray]:
    # Partition boxes
    data_1_id = partition_box(data_1)

    # Pair counting. NOTE: data_1 and data_2 must have the same dtype.
    dd_pairs = process_DD_pairs(
        data_1=data_1,
        data_2=data_2,
        data_1_id=data_1_id,
        radial_edges=radial_edges,
        weights_1=weights_1,
        weights_2=weights_2,
        nthreads=nthreads,
    )
    # Estimate jackknife samples of the tpcf
    xi, xi_i, mean_xi_i, cov = tpck_with_jk_from_DD(
        data_1=data_1,
        data_2=data_2,
        data_1_id=data_1_id,
        radial_edges=radial_edges,
        dd_pairs=dd_pairs,
    )

    # Total correlation function (measured once)
    # Correlation function per subsample (subvolume)
    # Mean correlation function (sample mean)
    # Covariance
    if jk_estimates:
        return xi, xi_i, mean_xi_i, cov
    # Mean correlation function
    # Covariance
    else:
        return mean_xi_i, cov


if __name__ == "__main__":
    pass
