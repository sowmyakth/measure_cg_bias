"""Unit test for cg bias code"""
import galsim
import numpy as np
import cg_functions as cg_fn
# galsim.ChromaticConvolution.resize_effective_prof_cache(5)


def no_bulge():
    """No bulge, cg bias must be zero"""
    bfrac = 1e-29
    in_p = cg_fn.LSST_Args(bulge_frac=bfrac)
    filt = galsim.Bandpass('data/baseline/total_r.dat',
                           wave_type='nm').thin(rel_err=1e-4)
    in_p.b_SED, in_p.d_SED, in_p.c_SED = cg_fn.get_template_seds(in_p)
    gal = cg_fn.get_gal_cg(in_p)
    meas_args = cg_fn.meas_args()
    meas_args.bp = filt
    psf_args = cg_fn.psf_params()
    gcg, gnocg = cg_fn.calc_cg_crg(gal, meas_args, psf_args)
    np.testing.assert_array_almost_equal((gcg / gnocg - 1).T[0], [0, 0],
                                         decimal=5)


def no_disk():
    """No disk, cg bias must be zero"""
    bfrac = 0.999999
    in_p = cg_fn.LSST_Args(bulge_frac=bfrac)
    filt = galsim.Bandpass('data/baseline/total_r.dat',
                           wave_type='nm').thin(rel_err=1e-4)
    in_p.b_SED, in_p.d_SED, in_p.c_SED = cg_fn.get_template_seds(in_p)
    gal = cg_fn.get_gal_cg(in_p)
    meas_args = cg_fn.meas_args()
    meas_args.bp = filt
    psf_args = cg_fn.psf_params()
    gcg, gnocg = cg_fn.calc_cg_crg(gal, meas_args, psf_args)
    np.testing.assert_array_almost_equal((gcg / gnocg - 1).T[0], [0, 0],
                                         decimal=5)


def same_bulge_disk():
    """Bulge and disk have same profile, cg bias must be zero"""
    in_p = cg_fn.LSST_Args(bulge_n=1, bulge_HLR=1.2)
    filt = galsim.Bandpass('data/baseline/total_r.dat',
                           wave_type='nm').thin(rel_err=1e-4)
    in_p.b_SED, in_p.d_SED, in_p.c_SED = cg_fn.get_template_seds(in_p)
    gal = cg_fn.get_gal_cg(in_p)
    meas_args = cg_fn.meas_args()
    meas_args.bp = filt
    psf_args = cg_fn.psf_params()
    gcg, gnocg = cg_fn.calc_cg_crg(gal, meas_args, psf_args)
    np.testing.assert_array_almost_equal((gcg / gnocg - 1).T[0], [0, 0],
                                         decimal=5)


def same_sed():
    """bulge and disk have same sed, cg bias must be zero"""
    in_p = cg_fn.LSST_Args(disk_SED_name='E')
    filt = galsim.Bandpass('data/baseline/total_r.dat',
                           wave_type='nm').thin(rel_err=1e-4)
    in_p.b_SED, in_p.d_SED, in_p.c_SED = cg_fn.get_template_seds(in_p)
    gal = cg_fn.get_gal_cg(in_p)
    meas_args = cg_fn.meas_args()
    meas_args.bp = filt
    psf_args = cg_fn.psf_params()
    gcg, gnocg = cg_fn.calc_cg_crg(gal, meas_args, psf_args)
    np.testing.assert_array_almost_equal((gcg / gnocg - 1).T[0], [0, 0],
                                         decimal=5)


def achr_psf():
    """LSST PSF is chromatic, CG bias must be 0"""
    in_p = cg_fn.LSST_Args()
    filt = galsim.Bandpass('data/baseline/total_r.dat',
                           wave_type='nm').thin(rel_err=1e-4)
    in_p.b_SED, in_p.d_SED, in_p.c_SED = cg_fn.get_template_seds(in_p)
    gal = cg_fn.get_gal_cg(in_p)
    meas_args = cg_fn.meas_args()
    meas_args.bp = filt
    psf_args = cg_fn.psf_params(alpha=0)
    gcg, gnocg = cg_fn.calc_cg_crg(gal, meas_args, psf_args)
    np.testing.assert_array_almost_equal((gcg / gnocg - 1).T[0],
                                         [0, 0],
                                         decimal=5)


def with_cg():
    """Raise error if no cg bias in default setting"""
    in_p = cg_fn.LSST_Args()
    filt = galsim.Bandpass('data/baseline/total_r.dat',
                           wave_type='nm').thin(rel_err=1e-4)
    in_p.b_SED, in_p.d_SED, in_p.c_SED = cg_fn.get_template_seds(in_p)
    gal = cg_fn.get_gal_cg(in_p)
    meas_args = cg_fn.meas_args()
    meas_args.bp = filt
    psf_args = cg_fn.psf_params()
    gcg, gnocg = cg_fn.calc_cg_crg(gal, meas_args, psf_args)
    np.testing.assert_raises(AssertionError,
                             np.testing.assert_array_almost_equal,
                             (gcg / gnocg - 1).T[0], [0, 0], decimal=5)
    # Previously measured value at default galaxy parameters
    np.testing.assert_array_almost_equal((gcg / gnocg - 1).T[0],
                                         [0.00106263, 0.00106594],
                                         decimal=5)


def cg_other_est():
    """Test all HSM shear estimators give same value as REGAUSS (default)"""
    in_p = cg_fn.LSST_Args()
    filt = galsim.Bandpass('data/baseline/total_r.dat',
                           wave_type='nm').thin(rel_err=1e-4)
    in_p.b_SED, in_p.d_SED, in_p.c_SED = cg_fn.get_template_seds(in_p)
    gal = cg_fn.get_gal_cg(in_p)
    psf_args = cg_fn.psf_params()
    meas_args = cg_fn.meas_args(shear_est='KSB')
    meas_args.bp = filt
    gcg, gnocg = cg_fn.calc_cg_crg(gal, meas_args, psf_args)
    # Previously measured value at default galaxy parameters
    np.testing.assert_array_almost_equal((gcg / gnocg - 1).T[0],
                                         [0.00106263, 0.00106594],
                                         decimal=5)
    meas_args.shear_est = 'BJ'
    gcg, gnocg = cg_fn.calc_cg_crg(gal, meas_args, psf_args)
    # Previously measured value at default galaxy parameters
    np.testing.assert_array_almost_equal((gcg / gnocg - 1).T[0],
                                         [0.00106263, 0.00106594],
                                         decimal=5)
    meas_args.shear_est = 'LINEAR'
    gcg, gnocg = cg_fn.calc_cg_crg(gal, meas_args, psf_args)
    # Previously measured value at default galaxy parameters
    np.testing.assert_array_almost_equal((gcg / gnocg - 1).T[0],
                                         [0.00106263, 0.00106594],
                                         decimal=5)


def with_CRG():
    """Raise error if no cg bias in default setting"""
    in_p = cg_fn.LSST_Args()
    filt = galsim.Bandpass('data/baseline/total_r.dat',
                           wave_type='nm').thin(rel_err=1e-4)
    in_p.b_SED, in_p.d_SED, in_p.c_SED = cg_fn.get_template_seds(in_p)
    gal = cg_fn.get_gal_cg(in_p)
    meas_args = cg_fn.meas_args()
    meas_args.bp = filt
    psf_args = cg_fn.psf_params()
    CRG1, CRG2 = cg_fn.get_CRG_basic(gal, in_p)
    gcg1, gnocg1 = cg_fn.calc_cg_crg(CRG1, meas_args, psf_args)
    gcg2, gnocg2 = cg_fn.calc_cg_crg(CRG2, meas_args, psf_args)
    # Previously measured value at default galaxy parameters
    np.testing.assert_array_almost_equal((gcg1 / gnocg1 - 1).T[0],
                                         [0.00128506, 0.0012862],
                                         decimal=5)
    np.testing.assert_array_almost_equal((gcg2 / gnocg2 - 1).T[0],
                                         [0.00106303, 0.00106446],
                                         decimal=5)


if __name__ == "__main__":
    no_bulge()
    no_disk()
    same_bulge_disk()
    same_sed()
    achr_psf()
    with_cg()
    cg_other_est()
    with_CRG()
    print "All Good!"
