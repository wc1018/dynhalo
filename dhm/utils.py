# -*- coding: utf-8 -*-
""" Some utility routines and constants
"""
import os
from dataclasses import dataclass
from datetime import timedelta
from time import time
from typing import Callable

__all__ = ["timer, mkdir"]


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


if __name__ == '__main__':
    pass
