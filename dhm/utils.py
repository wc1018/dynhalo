
from typing import Callable, Union
from datetime import timedelta
from time import time

import os
import numpy


class COLS:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    BULLET = "\u25CF"


OKGOOD = f"{COLS.OKGREEN}{COLS.BULLET}{COLS.ENDC} "
FAIL = f"{COLS.FAIL}{COLS.BULLET}{COLS.ENDC} "


def timer(
    procedure: Callable,
) -> Callable:
    """Decorator that prints the procedure's execution time."""

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


def mkdir(path, verbose=False):
    if verbose:
        print(path)
    if not os.path.exists(os.path.abspath(path)):
        os.mkdir(os.path.abspath(path))
    if verbose:
        print(f"Directory created at {os.path.exists(os.path.abspath(path))}")
    return