from typing import Callable, Union
from datetime import timedelta
from time import time
import numpy as np


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


def replace_val(var_io: np.ndarray,
                idx: Union[np.ndarray, list],
                value: Union[int, float],
                ) -> np.ndarray:
    '''Changes the values of 'var_io' by placing 'value' in all elements of 
    'idx'

    Args:
        var_io (np.ndarray): input-output array that will have elements
                             overwritten.
        idx (np.ndarray | list): mask with indices or boolean values.
        value (int | float): numerical value to write in all selected elements.

    Returns:
        np.ndarray: Returns the same array (might be a copy of) with new values.
    '''
    var_io[idx] = value
    return var_io
