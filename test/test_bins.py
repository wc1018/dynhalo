import numpy
import pytest

from dhm.corrfunc import bins
from dhm.utils import gen_data_pos_regular


def test_generate_bins():
    rmin = 0
    rmax = 5
    n = 5

    # Check default values. Since bmin=0 and soft=0, bins should NOT be
    # log-spaced.
    rbins, redges = bins.generate_bins(
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
    rbins, redges = bins.generate_bins(
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
            bins.generate_bin_str(item)

    assert bins.generate_bin_str([0, 1]) == '0.00-1.00'
    assert bins.generate_bin_str((0, 1)) == '0.00-1.00'


def test_partition_box():
    boxsize = 100
    grid_coarse = boxsize//4
    nside = numpy.int_(numpy.ceil(boxsize / grid_coarse))
    data_pos = gen_data_pos_regular(boxsize, grid_coarse)

    sorted_data = bins.partition_box(
        data=data_pos,
        boxsize=boxsize,
        gridsize=grid_coarse
    )

    assert len(sorted_data) == nside**3
    for item in sorted_data:
        assert len(item) == 1

    # Duplicate data_pos and test
    data_pos_2 = numpy.vstack([data_pos, data_pos])
    sorted_data = bins.partition_box(
        data=data_pos_2,
        boxsize=boxsize,
        gridsize=grid_coarse
    )

    assert len(sorted_data) == nside**3
    for item in sorted_data:
        assert len(item) == 2
