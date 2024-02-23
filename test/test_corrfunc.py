from typing import List

import h5py
import matplotlib.pyplot as plt
import numpy
import pytest

from dynhalo.corrfunc import bins, tpcf
from dynhalo.utils import gen_data_pos_random


def radial_bins(rmin=5, rmax=50, n=10):
    # Define radial bins to count pairs
    return bins.generate_bins(
        bmin=rmin,
        bmax=rmax,
        nbins=n,
    )


def ddpairs(radial_bins, boxsize, gridsize):
    # Generate synthetic data
    data_pos = gen_data_pos_random(boxsize, 10_000)
    sorted_data = bins.partition_box(
        data=data_pos,
        boxsize=boxsize,
        gridsize=gridsize
    )

    # Count pairs
    redges = radial_bins[-1]
    pairs = tpcf.process_DD_pairs(
        data_1=data_pos,
        data_2=data_pos,
        data_1_id=sorted_data,
        radial_edges=redges,
        boxsize=boxsize,
        gridsize=gridsize,
        nthreads=16,
    )
    return data_pos, sorted_data, redges, pairs


class TestTPCF():

    boxsize = 100
    # grid_fine = boxsize // 20
    grid_coarse = boxsize // 4

    dd_pairs_fixture = ddpairs(radial_bins(), boxsize, grid_coarse)

    def test_process_DD_pairs(self):
        #  Generate synthetic data
        nside = numpy.int_(numpy.ceil(self.boxsize / self.grid_coarse))
        _, _, _, dd_pairs = self.dd_pairs_fixture

        assert dd_pairs.shape == (nside**3, 10)

    def test_tpcf_jk(self):

        xi, xi_samples, xi_mean, xi_cov = tpcf.tpcf_jk(
            n_obj_d1=numpy.size(self.dd_pairs_fixture[0], 0),
            n_obj_d2=numpy.size(self.dd_pairs_fixture[0], 0),
            data_1_id=self.dd_pairs_fixture[1],
            radial_edges=self.dd_pairs_fixture[2],
            dd_pairs=self.dd_pairs_fixture[-1],
            boxsize=self.boxsize,
            gridsize=self.grid_coarse,
        )

        assert xi.size == 10
        assert xi_mean.size == 10
        assert xi_samples.shape == (64, 10)
        assert xi_cov.shape == (10, 10)

        assert numpy.all(xi < 1) and numpy.all(xi_mean < 1)
        assert numpy.all(numpy.diag(xi_cov) > 0)
        assert numpy.isclose(numpy.linalg.det(xi_cov), 0)


class TestTPCFResult:

    def test_halo_tpcf(self):

        test_data_file = 'test/data/test_data_corr_func.hdf5'
        with h5py.File(test_data_file, 'r') as hdf:
            res_xi = {}
            for key in hdf['sample_xi'].keys():
                res_xi[key] = hdf['sample_xi/'+key][()]
            halo_pos = hdf['sample_box/halo'][()]
            part_pos = hdf['sample_box/part'][()]

        rbins_fid = res_xi['r_bins']
        xi_fid = res_xi['xi']
        xi_cov_fid = res_xi['xi_cov']

        xi, _, _, xi_cov = tpcf.cross_tpcf_jk(
            data_1=halo_pos,
            data_2=part_pos,
            radial_edges=res_xi['r_edges'],
            boxsize=1_000,
            gridsize=250,
            nthreads=16,
        )

        plt.title('Test Correlation Function')
        _, axes = plt.subplots(2, 1, figsize=(3, 4),
                               gridspec_kw={'hspace': 0})
        plt.sca(axes[0])
        plt.errorbar(rbins_fid, rbins_fid**2*xi_fid,
                     yerr=rbins_fid**2*numpy.sqrt(numpy.diag(xi_cov_fid)),
                     fmt='.', elinewidth=1, capsize=3, color='k', alpha=0.5,
                     label='Fiducial')
        plt.errorbar(rbins_fid, rbins_fid**2*xi,
                     yerr=rbins_fid**2*numpy.sqrt(numpy.diag(xi_cov)),
                     fmt='.', elinewidth=1, capsize=3, color='r', alpha=0.5,
                     label='New')
        plt.ylabel(r'$r^2\xi_{hm}$')
        plt.legend(loc='upper right')
        plt.xscale('log')

        plt.sca(axes[1])
        plt.plot(rbins_fid, xi/xi_fid, lw=1, color='r', alpha=0.5)
        plt.hlines(1, 0.9*min(rbins_fid), 1.1*max(rbins_fid), color='k',
                   ls=':', lw=1)
        plt.xscale('log')
        plt.xlabel(r'$r~[{\rm Mpc}/h]$')
        plt.ylabel(r'$\xi/\xi_{\rm fid}$')
        plt.tight_layout()
        plt.savefig('test/img/test_sample_box_xi.png', dpi=150)
