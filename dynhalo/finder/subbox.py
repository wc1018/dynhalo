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
        Cartesian coordinates
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
        low = chunk * chunksize
        if chunk < n_iter - 2:
            upp = (chunk + 1) * chunksize
        else:
            upp = None
        ids[low:upp] = get_sub_box_id(positions[low:upp], boxsize, subsize)

    if name:
        file_name = f'sub_box_id_{name}.hdf5'
    else:
        file_name = f'sub_box_id.hdf5'
    with h5.File(path + file_name, 'w') as hdf:
        hdf.create_dataset('SBID', data=ids, dtype=uint_dtype)

    return None


@timer
def split_simulation_into_sub_boxes(
    positions: np.ndarray,
    velocities: np.ndarray,
    ids: np.ndarray,
    boxsize: float,
    subsize: float,
    chunksize: float,
    dtypes: list,
    path: str,
    name: str = None,
) -> None:
    # Check chunksize is smaller than the number of items
    n_items = positions.shape[0]
    if chunksize > n_items:
        raise ValueError(
            f"The specified chunksize {chunksize} is larger than the number of items {n_items}")

    # Create directory if it does not exist
    save_path = path + 'sub_boxes/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not os.path.exists(save_path):
        generate_sub_box_ids(positions, boxsize, subsize, chunksize, path, name)
    
    if name:
        sub_box_ids_file = path + f'sub_box_id_{name}.hdf5'
    else:
        sub_box_ids_file = path + f'sub_box_id.hdf5'
    with h5.File(sub_box_ids_file, 'r') as hdf:
        sub_box_ids = hdf['SBID'][()]
    
    n_iter = n_items // chunksize

    uint_dtype_row = get_np_unit_dytpe(n_items)
    row_idx = np.arange(n_items, dtype=uint_dtype_row)

    uint_dtype_ids = get_np_unit_dytpe(np.max(ids))
    for chunk in tqdm(range(n_iter), desc='Chunk', ncols=100, colour='blue'):
        # Select chunk
        low = chunk * chunksize
        if chunk < n_iter - 2:
            upp = (chunk + 1) * chunksize
        else:
            upp = None
        sb_ids = sub_box_ids[low:upp]
        pos = positions[low:upp]
        vel = velocities[low:upp]
        pid = ids[low:upp]
        row = row_idx[low:upp]

        sb_unique = np.unique(sb_ids)
        order = np.argsort(sb_ids)
        for item in sb_unique:
            left = np.searchsorted(sb_ids, item, side="left", sorter=order)
            right = np.searchsorted(sb_ids, item, side="right", sorter=order)

            pos_item = pos[order][left:right]
            vel_item = vel[order][left:right]
            pid_item = pid[order][left:right]
            row_item = row[order][left:right]

            with h5.File(save_path + f"{item}.hdf5", "a") as hdf:
                if not name in hdf.keys():
                    hdf.create_group(name)
                
                # If it is the first time opening this file, create datasets
                if not 'ID' in hdf[name].keys():
                    hdf.create_dataset(
                        name=f'{name}/ID',
                        data=pid_item,
                        maxshape=(None, ),
                        dtype=uint_dtype_ids,
                    )
                    hdf.create_dataset(
                        name=f'{name}/pos',
                        data=pos_item,
                        maxshape=(None, pos_item.shape[-1]),
                        dtype=dtypes[0],
                    )
                    hdf.create_dataset(
                        name=f'{name}/vel',
                        data=vel_item,
                        maxshape=(None, vel_item.shape[-1]),
                        dtype=dtypes[1],
                    )
                    hdf.create_dataset(
                        name=f'{name}/row_idx',
                        data=row_item,
                        maxshape=(None, ),
                        dtype=uint_dtype_row,
                    )
                # If it is not the first time opening the file, reshape the 
                # datasets
                else:
                    last_item = pid_item.shape[0]
                    new_shape = hdf[f'{name}/ID'].shape[0] + last_item

                    hdf[f'{name}/ID'].resize((new_shape), axis=0)
                    hdf[f'{name}/ID'][-last_item:] = pid_item
                    hdf[f'{name}/pos'].resize((new_shape), axis=0)
                    hdf[f'{name}/pos'][-last_item:] = pos_item
                    hdf[f'{name}/vel'].resize((new_shape), axis=0)
                    hdf[f'{name}/vel'][-last_item:] = vel_item
                    hdf[f'{name}/row_idx'].resize((new_shape), axis=0)
                    hdf[f'{name}/row_idx'][-last_item:] = row_item

    return None


if __name__ == '__main__':
    pass
