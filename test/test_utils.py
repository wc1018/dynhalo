import os

import numpy
import pytest

from dhm import utils


def test_mkdir():
    dirpath_fail = '/zzz/tmp_dir_test/'
    with pytest.raises(FileNotFoundError):
        utils.mkdir(dirpath_fail, verbose=False)

    dirpath_pass = os.getcwd() + '/test/tmp_dir_test/'
    utils.mkdir(dirpath_pass, verbose=False)
    assert os.path.exists(dirpath_pass)
    os.removedirs(dirpath_pass)


def test_cartesian_product():
    # List [0, 1, 2]
    n_points = 3
    points = numpy.arange(n_points)
    points_float = numpy.linspace(0, n_points-1, n_points)

    # Repeat the list twice and thrice
    arrs_2 = 2 * [points]
    arrs_3 = 3 * [points]

    cart_prod_1 = utils.cartesian_product([points])
    cart_prod_2 = utils.cartesian_product(arrs_2)
    cart_prod_3 = utils.cartesian_product(arrs_3)
    cart_prod_float = utils.cartesian_product([points, points_float])

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
