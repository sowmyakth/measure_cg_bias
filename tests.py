"""Unit test for cg bias code"""
import galsim
import numpy as np
import cg_functions as cg_fn
galsim.ChromaticConvolution.resize_effective_prof_cache(5)


def no_bulge():
    """No bulge, cg bias must be zero"""
    bfrac = 1e-29
    in_p = cg_fn.LSST_Args(bulge_frac=bfrac)
    filt = galsim.Bandpass('data/baseline/total_r.dat',
                           wave_type='nm').thin(rel_err=1e-4)
    waves = np.linspace(10, 900, 2)
    b_SED = galsim.SED("data/CWW_E_ext.sed", flux_type='flambda', 
                       wave_type='Ang')
    d_SED = galsim.SED("data/CWW_Im_ext.sed", flux_type='flambda', 
                       wave_type='Ang')
    c_SED = bfrac * b_SED + (1 - bfrac) * d_SED
    in_p.b_SED, in_p.d_SED, in_p.c_SED = b_SED, d_SED, c_SED
    gal = cg_fn.get_gal_cg(in_p)
    meas_args = cg_fn.meas_args()
    meas_args.bp = filt
    psf_args = cg_fn.psf_params()
    gcg, gnocg = cg_fn.calc_cg_crg(gal, meas_args, psf_args)
    gtrue = meas_args.rt_g
    np.testing.assert_array_almost_equal((gcg / gnocg - 1).T[0], [0, 0],
                                         decimal=5)


def no_disk():
    """No disk, cg bias must be zero"""
    bfrac = 0.999999999999
    in_p = cg_fn.LSST_Args(bulge_frac=bfrac)
    filt = galsim.Bandpass('data/baseline/total_r.dat',
                           wave_type='nm').thin(rel_err=1e-4)
    waves = np.linspace(10, 900, 2)
    b_SED = galsim.SED("data/CWW_E_ext.sed", flux_type='flambda', 
                       wave_type='Ang')
    d_SED = galsim.SED("data/CWW_Im_ext.sed", flux_type='flambda', 
                       wave_type='Ang')
    c_SED = bfrac * b_SED + (1 - bfrac) * d_SED
    in_p.b_SED, in_p.d_SED, in_p.c_SED = b_SED, d_SED, c_SED
    gal = cg_fn.get_gal_cg(in_p)
    meas_args = cg_fn.meas_args()
    meas_args.bp = filt
    psf_args = cg_fn.psf_params()
    gcg, gnocg = cg_fn.calc_cg_crg(gal, meas_args, psf_args)
    gtrue = meas_args.rt_g
    np.testing.assert_array_almost_equal((gcg/gnocg-1).T[0], [0,0], decimal=5)


def same_bulge_disk():
    """Bulge and disk have same profile, cg bias must be zero"""
    in_p = cg_fn.LSST_Args(bulge_n=1, bulge_HLR=1.2)
    filt = galsim.Bandpass('data/baseline/total_r.dat',
                           wave_type='nm').thin(rel_err=1e-4)
    waves = np.linspace(10, 900, 2)
    b_SED = galsim.SED("data/CWW_E_ext.sed", flux_type='flambda', 
                       wave_type='Ang')
    d_SED = galsim.SED("data/CWW_Im_ext.sed", flux_type='flambda', 
                       wave_type='Ang')
    bfrac = in_p.bulge_frac
    c_SED = bfrac * b_SED + (1 - bfrac) * d_SED
    in_p.b_SED, in_p.d_SED, in_p.c_SED = b_SED, d_SED, c_SED
    gal = cg_fn.get_gal_cg(in_p)
    meas_args = cg_fn.meas_args()
    meas_args.bp = filt
    psf_args = cg_fn.psf_params()
    gcg, gnocg = cg_fn.calc_cg_crg(gal, meas_args, psf_args)
    gtrue = meas_args.rt_g
    np.testing.assert_array_almost_equal((gcg/gnocg-1).T[0], [0,0], decimal=5)


def same_sed():
    """bulge and disk have same sed, cg bias must be zero"""
    in_p = cg_fn.LSST_Args()
    filt = galsim.Bandpass('data/baseline/total_r.dat',
                           wave_type='nm').thin(rel_err=1e-4)
    waves = np.linspace(10, 900, 2)
    b_SED = galsim.SED("data/CWW_E_ext.sed", flux_type='flambda', 
                       wave_type='Ang')
    d_SED = galsim.SED("data/CWW_Im_ext.sed", flux_type='flambda', 
                       wave_type='Ang')
    bfrac = in_p.bulge_frac
    c_SED = bfrac * b_SED + (1 - bfrac) * d_SED
    in_p.b_SED, in_p.d_SED, in_p.c_SED = b_SED, b_SED, b_SED
    gal = cg_fn.get_gal_cg(in_p)
    meas_args = cg_fn.meas_args()
    meas_args.bp = filt
    psf_args = cg_fn.psf_params()
    gcg, gnocg = cg_fn.calc_cg_crg(gal, meas_args, psf_args)
    gtrue = meas_args.rt_g
    np.testing.assert_array_almost_equal((gcg/gnocg-1).T[0], [0,0], decimal=5)


def with_cg():
    """Raise error if no cg bias in default setting"""
    in_p = cg_fn.LSST_Args()
    filt = galsim.Bandpass('data/baseline/total_r.dat',
                           wave_type='nm').thin(rel_err=1e-4)
    waves = np.linspace(10, 900, 2)
    b_SED = galsim.SED("data/CWW_E_ext.sed", flux_type='flambda',
                       wave_type='Ang')
    d_SED = galsim.SED("data/CWW_Im_ext.sed", flux_type='flambda',
                       wave_type='Ang')
    bfrac = in_p.bulge_frac
    c_SED = bfrac * b_SED + (1 - bfrac) * d_SED
    in_p.b_SED, in_p.d_SED, in_p.c_SED = b_SED, d_SED, c_SED
    gal = cg_fn.get_gal_cg(in_p)
    meas_args = cg_fn.meas_args()
    meas_args.bp = filt
    psf_args = cg_fn.psf_params()
    gcg, gnocg = cg_fn.calc_cg_crg(gal, meas_args, psf_args)
    gtrue = meas_args.rt_g
    np.testing.assert_raises(AssertionError,
                             np.testing.assert_array_almost_equal,
                             (gcg / gnocg - 1).T[0], [0, 0], decimal=5)


if __name__ == "__main__":
    no_bulge()
    no_disk()
    same_bulge_disk()
    same_sed()
    with_cg()
    print "All Good!"
