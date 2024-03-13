import os

from typing import Tuple
import h5py as h5
import numpy as np
from tqdm import tqdm

from dynhalo.utils import timer, cartesian_product, get_np_unit_dytpe


def relative_coordinates(
    x0: np.ndarray,
    x: np.ndarray,
    boxsize: float,
    periodic: bool = True
) -> float:
    """Returns the coordinates x relative to x0 accounting for periodic boundary
    conditions

    Parameters
    ----------
    x0 : np.ndarray
        Reference position in cartesian coordinates
    x : np.ndarray
        Position array (N, 3)
    boxsize : float
        Size of simulation box
    periodic : bool, optional
        Set to True if the simulation box is periodic, by default True

    Returns
    -------
    float
        Relative positions
    """
    if periodic:
        return (x - x0 + 0.5*boxsize) % boxsize - 0.5*boxsize
    return x - x0


def generate_sub_box_grid(
    boxsize: float,
    subsize: float,
) -> Tuple[np.ndarray]:
    """Generates a 3D grid of sub-boxes.

    Parameters
    ----------
    boxsize : float
        Size of simulation box
    subsize : float
        Size of sub-box

    Returns
    -------
    Tuple[np.ndarray]
        ID and centre coordinate for all sub-boxes
    """

    # Number of sub-boxes per side
    boxes_per_side = np.int_(np.ceil(boxsize / subsize))

    # Determine data type for integer arrays based on the maximum number of 
    # elements
    uint_dtype = get_np_unit_dytpe(boxes_per_side)
    # Set of natural numbers from 0 to N-1
    n_range = np.arange(boxes_per_side, dtype=uint_dtype)

    # Shift in each dimension for numbering sub-boxes
    uint_dtype = get_np_unit_dytpe(boxes_per_side**2)
    shift = np.array(
        [1, boxes_per_side, boxes_per_side * boxes_per_side], dtype=uint_dtype)

    # Set of index vectors. Each vector points to the (i, j, k)-th sub-box
    n_pos = np.int_(cartesian_product([n_range, n_range, n_range]))

    # Set of all possible unique IDs for each sub-box
    ids = np.sum(n_pos * shift, axis=1)
    sort_order = np.argsort(ids)

    # Sort IDs so that the ID matches the row index.
    n_pos = n_pos[sort_order]
    ids = ids[sort_order]

    # Sub-box central coordinate. Populate each sub-box with one point at the
    # centre.
    centres = subsize * (n_pos + 0.5)
    return ids, centres


def get_sub_box_id(
    x: np.ndarray,
    boxsize: float,
    subsize: float,
) -> int:
    """Returns the sub-box ID to which the coordinates `x` fall into

    Parameters
    ----------
    x : np.ndarray
        Position in cartesian coordinates
    boxsize : float
        Size of simulation box
    subsize : float
        Size of sub-box

    Returns
    -------
    int
        ID of the sub-box
    """
    # Number of sub-boxes per side
    boxes_per_side = np.int_(np.ceil(boxsize / subsize))
    # Determine data type for integer arrays based on the maximum number of 
    # elements
    uint_dtype = get_np_unit_dytpe(boxes_per_side**2)
    # Shift in each dimension for numbering sub-boxes
    shift = np.array(
        [1, boxes_per_side, boxes_per_side * boxes_per_side], dtype=uint_dtype)
    return np.int_(np.sum(shift * np.floor(x / subsize), axis=1))


def get_adjacent_sub_box_ids(
    id: np.ndarray,
    ids: np.ndarray,
    positions: np.ndarray,
    boxsize: float,
    subsize: float,
) -> np.ndarray:
    """Returns a list of all IDs that are adjacent to the specified sub-box ID.
    There are always 27 adjacent boxes in a 3D volume.

    Parameters
    ----------
    id : np.ndarray
        ID of the sub-box
    ids : np.ndarray
        IDs of all sub-boxes
    positions : np.ndarray
        Positions of all the centres of the sub-boxes
    boxsize : float
        Size of simulation box
    subsize : float
        Size of sub-box

    Returns
    -------
    np.ndarray
        List of sub-box IDs adjacent to `id`

    Raises
    ------
    ValueError
        If `id` is not found in the allowed values in `ids`
    """
    if id not in ids:
        raise ValueError(f'ID {id} is out of bounds')

    x0 = positions[ids == id]
    d = relative_coordinates(x0, positions, boxsize)
    d = np.sqrt(np.sum(np.square(d), axis=1))
    mask = d <= 1.01*np.sqrt(3)*subsize
    return ids[mask]


@timer
def generate_sub_box_ids(
    positions: np.ndarray,
    boxsize: float,
    subsize: float,
    chunksize: float,
    path: str,
    name: str = None
) -> None:
    """Gets the sub-box ID for each position

    Parameters
    ----------
    positions : np.ndarray
        _description_
    boxsize : float
        Size of simulation box
    subsize : float
        Size of sub-box
    chunksize : float
        Number of items to process at a time in chunks
    path : str
        Where to save the IDs
    name : str, optional
        An additional name or identifier appended at the end of the file name, 
        by default None

    Returns
    -------
    None
    """
    n_items = positions.shape[0]
    n_iter = n_items // chunksize

    # Determine data type for integer arrays based on the maximum number of 
    # elements
    boxes_per_side = np.int_(np.ceil(boxsize / subsize))
    uint_dtype = get_np_unit_dytpe(boxes_per_side**3)

    ids = np.zeros(n_items, dtype=uint_dtype)

    for chunk in tqdm(range(n_iter), desc='Chunk', ncols=100, colour='blue'):
        if chunk < n_iter - 2:
            low = chunk * chunksize
            upp = (chunk + 1) * chunksize
            ids[low:upp] = get_sub_box_id(positions[low:upp], boxsize, subsize)
        else:
            low = chunk * chunksize
            ids[low:] = get_sub_box_id(positions[low:], boxsize, subsize)
    with h5.File(path+f'sub_box_id{name}.hdf5', 'w') as hdf:
        hdf.create_dataset('ID', data=ids, dtype=uint_dtype)

    return None


@timer
def split_simulation_into_sub_boxes(
    positions: np.ndarray,
    boxsize: float,
    subsize: float,
    chunksize: float,
    path: str,
) -> None:
    # Create directory if it does not exist
    if not os.path.exists(path + 'sub_boxes'):
        os.makedirs(path + 'sub_boxes')
    
    # uint_dtype = get_np_unit_dytpe(boxes_per_side**3)
    n_items = positions.shape[0]
    n_iter = n_items // chunksize

    
    return None

if __name__ == '__main__':
    pass
