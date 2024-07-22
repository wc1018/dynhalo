# -*- coding: utf-8 -*-
""" Some utility routines and constants
"""
import os
from dataclasses import dataclass
from datetime import timedelta
from time import time
from typing import Callable, List

import numpy

__all__ = ["timer, mkdir"]

# Gravitational constant
G_gravity = 4.3e-09     # Mpc (km/s)^2 / M_sun

@dataclass(frozen=True)
class COLS:
    """
    """
    HEADER: str = "\033[95m"
    OKBLUE: str = "\033[94m"
    OKCYAN: str = "\033[96m"
    OKGREEN: str = "\033[92m"
    WARNING: str = "\033[93m"
    FAIL: str = "\033[91m"
    ENDC: str = "\033[0m"
    BOLD: str = "\033[1m"
    UNDERLINE: str = "\033[4m"
    BULLET: str = "\u25CF"


OKGOOD = f"{COLS.OKGREEN}{COLS.BULLET}{COLS.ENDC} "
FAIL = f"{COLS.FAIL}{COLS.BULLET}{COLS.ENDC} "


def get_np_unit_dytpe(obj):
    np_unit_dtypes = numpy.array([numpy.uint16, numpy.uint32, numpy.uint64])
    loc = numpy.argmax(
        [obj < numpy.iinfo(item).max for item in np_unit_dtypes])
    return np_unit_dtypes[loc]


def timer(procedure: Callable) -> Callable:
    """Decorator that prints the procedure's execution time

    Parameters
    ----------
    procedure : Callable
        Any callable

    Returns
    -------
    Callable
        Returns callable object/return value
    """

    def wrapper(*args, **kwargs):
        start = time()
        return_value = procedure(*args, **kwargs)
        print(
            f"\t{COLS.BULLET}{COLS.BOLD}{COLS.WARNING} Elapsed time:{COLS.ENDC} "
            + f"{COLS.OKCYAN}{timedelta(seconds=time()-start)}{COLS.ENDC} "
            + f"{COLS.OKGREEN}{procedure.__name__}{COLS.ENDC}"
        )
        return return_value

    return wrapper


def mkdir(path: str, verbose: bool = False) -> None:
    """Checks if a path exists and creates a directory in path if not.

    Parameters
    ----------
    path : str
        Path where directory should exist. 
    verbose : bool, optional
        Whether to print info on directory creation process, by default False

    Returns
    -------
    None
    """
    abspath = os.path.abspath(path)
    isdir = os.path.isdir(abspath)
    if isdir:
        if verbose:
            print(f"Directory exists at {abspath}")
        return None
    else:
        print("Creating directory")
        try:
            os.mkdir(os.path.abspath(path))
            if verbose:
                print(f"Directory created at {abspath}")
        except:
            print(f"Directory could not be created at {abspath}")
            raise
    return None


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


def gen_data_pos_regular(boxsize, gridsize) -> numpy.ndarray:
    # Populate coordinates with one particle per subbox at the centre in steps
    # of nside between particles
    nside = numpy.int_(numpy.ceil(boxsize / gridsize))
    n_range = numpy.arange(nside, dtype=int)
    n_pos = numpy.int_(cartesian_product([n_range, n_range, n_range]))
    data_pos = gridsize * (n_pos + 0.5)
    return data_pos


def gen_data_pos_random(boxsize, nsamples) -> numpy.ndarray:
    data_pos = boxsize * numpy.random.uniform(0, 1, (nsamples, 3))
    return data_pos


if __name__ == '__main__':
    pass
