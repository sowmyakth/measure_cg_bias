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


def get_CRG(gal, in_p):
    """get CRG for reference galaxy"""
    hst_param = cg_fn.Eu_Args(scale=0.03, psf_sig_o=0.071,
                              psf_w_o=806)
    PSF = cg_fn.get_gaussian_PSF(hst_param)
    con = galsim.Convolve([gal, PSF])
    # get bandpass
    V_band = cg_fn.get_HST_Bandpass('F606W')
    I_band = cg_fn.get_HST_Bandpass('F814W')
    xi_v = galsim.getCOSMOSNoise(file_name='data/acs_V_unrot_sci_cf.fits',
                                 variance=1e-39)
    xi_i = galsim.getCOSMOSNoise(file_name='data/acs_I_unrot_sci_cf.fits',
                                 variance=1e-39)
    psf_v = cg_fn.get_eff_psf(PSF, in_p.c_SED, V_band)
    psf_i = cg_fn.get_eff_psf(PSF, in_p.c_SED, I_band)
    eff_PSFs = [psf_v, psf_i]
    gal_im_v = con.drawImage(V_band, nx=350, ny=350, scale=0.03)
    gal_im_i = con.drawImage(I_band, nx=350, ny=350, scale=0.03)
    #  Polynomial SEDs
    images = [gal_im_v, gal_im_i]
    crg1 = galsim.ChromaticRealGalaxy.makeFromImages(images=images,
                                                     bands=[V_band, I_band],
                                                     xis=[xi_v, xi_i],
                                                     PSFs=eff_PSFs)
    #  True SEDs
    seds = [in_p.b_SED, in_p.d_SED]
    crg2 = galsim.ChromaticRealGalaxy.makeFromImages(images=images,
                                                     bands=[V_band, I_band],
                                                     xis=[xi_v, xi_i],
                                                     PSFs=eff_PSFs,
                                                     SEDs=seds)
    return crg1, crg2


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
    CRG1, CRG2 = get_CRG(gal, in_p)
    gcg1, gnocg1 = cg_fn.calc_cg_crg(CRG1, meas_args, psf_args)
    gcg2, gnocg2 = cg_fn.calc_cg_crg(CRG2, meas_args, psf_args)
    # Previously measured value at default galaxy parameters
    np.testing.assert_array_almost_equal((gcg2 / gnocg2 - 1).T[0],
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
    with_CRG()
    print "All Good!"
