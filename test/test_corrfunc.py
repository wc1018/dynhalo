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
