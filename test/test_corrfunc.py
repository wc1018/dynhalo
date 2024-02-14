from typing import List

import numpy
import pytest

from dhm.corrfunc import tpcf


def test_generate_bins():
    rmin = 0
    rmax = 5
    n = 5

    # Check default values. Since bmin=0 and soft=0, bins should NOT be
    # log-spaced.
    rbins, redges = tpcf.generate_bins(
        bmin=rmin,
        bmax=rmax,
        nbins=n,
    )

    assert len(rbins) == 5
    assert len(redges) == 6
    assert rbins[0] == 0.5
    assert redges[0] == 0
    assert rbins[-1] == 4.5
    assert redges[-1] == 5

    # Introduce a softening with log-spaced bins
    soft = 0.5
    rbins, redges = tpcf.generate_bins(
        bmin=rmin,
        bmax=rmax,
        nbins=n,
        soft=soft
    )

    assert len(rbins) == 5
    assert len(redges) == 6
    assert rbins[0] == pytest.approx(0.646223)
    assert redges[0] == 0.5
    assert rbins[-1] == pytest.approx(4.077393)
    assert redges[-1] == pytest.approx(5)  # Precision issues with 5


def test_generate_bin_str():
    # Check for correct input type
    for item in [0, 1., -1., "a", 4+1j, None]:
        with pytest.raises(ValueError):
            tpcf.generate_bin_str(item)

    assert tpcf.generate_bin_str([0, 1]) == '0.00-1.00'
    assert tpcf.generate_bin_str((0, 1)) == '0.00-1.00'


def cartesian_product(arrays: List[numpy.ndarray]):
    """Generalized N-dimensional products
    Taken from https://stackoverflow.com/questions/11144513/
    Answer by Nico Schlömer
    Updated for numpy > 1.25

    Parameters
    ----------
    arrays : List[numpy.ndarray]
        _description_

    Returns
    -------
    _type_
        _description_
    """
    la = len(arrays)
    dtype = numpy.result_type(*[a.dtype for a in arrays])
    arr = numpy.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(numpy.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def test_cartesian_product():
    # List [0, 1, 2]
    n_points = 3
    points = numpy.arange(n_points)
    points_float = numpy.linspace(0, n_points-1, n_points)

    # Repeat the list twice and thrice
    arrs_2 = 2 * [points]
    arrs_3 = 3 * [points]

    cart_prod_1 = cartesian_product([points])
    cart_prod_2 = cartesian_product(arrs_2)
    cart_prod_3 = cartesian_product(arrs_3)
    cart_prod_float = cartesian_product([points, points_float])

    # Cardinality of Nx...xN = N^n
    assert len(cart_prod_1) == n_points
    assert len(cart_prod_2) == n_points*n_points
    assert len(cart_prod_3) == n_points*n_points*n_points
    # Each element has shape (n,)
    assert cart_prod_1[0].shape == (1,)
    assert cart_prod_2[0].shape == (2,)
    assert cart_prod_3[0].shape == (3,)
    # Check dtypes
    assert type(cart_prod_1[0][0]) == numpy.int64
    assert type(cart_prod_float[0][0]) == numpy.float64


def test_partition_box():
    boxsize = 100
    gridsize = 20
    # Populate coordinates with one particle per subbox at the centre in steps
    # of nside between particles
    nside = numpy.int_(numpy.ceil(boxsize / gridsize))
    n_range = numpy.arange(nside, dtype=int)
    n_pos = numpy.int_(cartesian_product([n_range, n_range, n_range]))
    data_pos = gridsize * (n_pos + 0.5)

    sorted_data = tpcf.partition_box(
        data=data_pos,
        boxsize=boxsize, 
        gridsize=gridsize
    )

    assert len(sorted_data) == nside**3
