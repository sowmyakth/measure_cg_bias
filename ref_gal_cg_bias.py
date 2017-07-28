"""Code to measure bias from color gradients for reference galaxy. All
measurements are noise free.
Galaxy parametrs are the default values in cg_functions. Bias is measured for
1) parametric reference galaxy
2) para gal > CRG with polynomial SED
3) para gal -> CRG with true SED

1) Draw reference parametric bulge + disk galaxy as seen by LSST and measure
cg bias for different redshifts. Bias is also measured for the reference
galaxy with two different disk SEDs.

2) Galaxies in 2 are drawn as seen by HST i.e paramteric galaxy is convolved
with HST PSF and seen in HST bands. The "HST" images are input to
galasim.ChromaticRealGalaxy(CRG) which produces the a pre convolution surface
brighness profile of the galaxy that preserves its chromatic features. CG
bias for this galaxy when seen by LSST is estimated. CRG computes chromatic
SBP with polynomial SEDs which is the default setting.

3) Same as 2 but with CRG using true SEDs for galaxy SBP.
"""
import galsim
import numpy as np
from astropy.table import Table
import cg_functions as cg_fn
# galsim.ChromaticConvolution.resize_effective_prof_cache(5)


def get_table(num):
    names = ('NUMBER', 'redshift', 'para_g_cg', 'para_g_no_cg',
             'CRG_g_cg', 'CRG_g_no_cg', 'CRG_tru_g_cg', 'CRG_tru_g_no_cg',
             'para_m1', 'CRG_m1', 'CRG_tru_m1', 'para_m2', 'CRG_m2',
             'CRG_tru_m2', 'para_c1', 'CRG_c1', 'CRG_tru_c1', 'para_c2',
             'CRG_c2', 'CRG_tru_c2')
    dtype = ['int'] + ['float'] * 19
    cols = [range(num), np.zeros(num)] + [np.zeros([num, 2, 2])] * 6 + [np.zeros([num, 3])] * 12
    index_table = Table(cols, names=names, dtype=dtype)
    return index_table


def get_CRG(in_p):
    """Create CRG for a given input galaxy parametrs.
    Bulge + Disk galaxy is created, convolved with HST PSF, drawn in HST V and
    I bands for 1 second exposure. Correlated noise (from AEGIS images)
    is added to each image. SNR in a gaussian elliptical aperture is computed.
    cgr1: The galaxy +psf images + noise correlation function is provided as
    input to CRG with default polynomial SEDs. The input galaxy images are
    padded with noise. This enables us to to draw the CRG image larger than
    the input image, and not have boundary edges.
    crg2: same as crg1 except the true SEDS of bulge and disk are provided
    as input to CRG.
    """
    #  HST scale
    scale = 0.03
    area = 4.437 * 10000  # np.pi * (2.4 * 100 / 2.)**2
    v_exptime = 1  # 2260
    i_exptime = 1  # 2100
    nx, ny = 360, 360   # number of x and y pixels
    in_p.b_SED, in_p.d_SED, in_p.c_SED = cg_fn.get_template_seds(in_p)
    PSF = cg_fn.get_gaussian_PSF(in_p)
    gal = cg_fn.get_gal_cg(in_p)
    con = galsim.Convolve([gal, PSF])
    # get bandpass
    V_band = cg_fn.get_HST_Bandpass('F606W')
    I_band = cg_fn.get_HST_Bandpass('F814W')
    gal_im_v = con.drawImage(V_band, nx=nx, ny=ny, scale=scale,
                             area=area, exptime=v_exptime)
    gal_im_i = con.drawImage(I_band, nx=nx, ny=ny, scale=scale,
                             area=area, exptime=i_exptime)
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


def get_lsst_para(in_p):
    """Create parametric bulge+disk galaxy for input parametrs."""
    in_p.b_SED, in_p.d_SED, in_p.c_SED = cg_fn.get_template_seds(in_p)
    gal = cg_fn.get_gal_cg(in_p)
    return gal


def meas_cg_bias(gal, row, f_name,
                 rt_g, f_type, npix=360):
    """Computes bias due to color gradient on sahpe measuremnt.
    For an input chromatic galaxy with cg,  gal an equilvalent galaxy with
    no cg is created and the shear recovered from each (with ring test) is
    measured.
    @gal     input galaxy with cg.
    @row     astropy table row to save measured shear.
    @f_name  name of filter to measure cg bias in
    @rt_g    shaer applied to the galaxy.
    @type    string to identify the column of row to save measured shear.
    """
    print " Measuring CG bias"
    filt = galsim.Bandpass('data/baseline/total_%s.dat'%f_name,
                           wave_type='nm').thin(rel_err=1e-4)
    meas_args = cg_fn.meas_args(rt_g=rt_g, npix=npix)
    meas_args.bp = filt
    psf_args = cg_fn.psf_params()
    gcg, gnocg = cg_fn.calc_cg_crg(gal, meas_args, psf_args)
    print " Measured CG bias"
    if (gcg == "Fail") or (gnocg == "Fail"):
        print "HSM FAILED"
        g_f = np.ones([2, len(rt_g)]) * -10
        gcg, gnocg = g_f, g_f
    row[f_type + '_g_cg'] = gcg.T
    row[f_type + '_g_no_cg'] = gnocg.T
    m, c = cg_fn.get_bias(gcg.T[0], gnocg.T[0], rt_g.T[0])
    row[f_type + '_m1'] = m
    row[f_type + '_c1'] = c
    m, c = cg_fn.get_bias(gcg.T[1], gnocg.T[1], rt_g.T[1])
    row[f_type + '_m2'] = m
    row[f_type + '_c2'] = c


def main(Args):
    """Creates galaxy and psf images with different parametrs
     and measures cg bias"""
    # Set disk SED name
    dSEDs = ['Im', 'Sbc', 'Scd']
    if Args.disk_SED_name != 'all':
        dSEDs = [Args.disk_SED_name, ]
    dSED = Args.disk_SED_name
    redshifts = np.linspace(0., 1.2, 5)
    e_s = [0.3, 0.3]
    filt = Args.filter
    g = np.linspace(0.005, 0.01, 2)
    rt_g = np.array([g, g]).T
    num = len(redshifts)
    for d, dSED in enumerate(dSEDs):
        for z_num, z in enumerate(redshifts):
            print "Creating gal at redshift {0} in {1} band".format(z, filt)
            index_table = get_table(num)
            input_p1 = cg_fn.Eu_Args(scale=0.03, redshift=z,
                                     bulge_e=e_s, disk_e=e_s,
                                     psf_sig_o=0.071, psf_w_o=806,
                                     disk_SED_name=dSED)
            CRG1, CRG2 = get_CRG(input_p1)
            meas_cg_bias(CRG1, index_table[z_num], filt,
                         rt_g, 'CRG')
            meas_cg_bias(CRG2, index_table[z_num], filt,
                         rt_g, 'CRG_tru')
            # parametric
            input_p2 = cg_fn.LSST_Args(disk_SED_name=dSED, redshift=z)
            para_gal = get_lsst_para(input_p2)
            meas_cg_bias(para_gal, index_table[z_num], filt,
                         rt_g, 'para')
        op_file = 'results/ref_gal_cg_bias_{0}_dsed_{1}_band.fits'.format(dSED,
                                                                         filt)
        index_table.write(op_file, format='fits',
                          overwrite=True)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--disk_SED_name', default='all',
                        help="run for only one disk SED.[Default:all]")
    parser.add_argument('--filter', default='r',
                        help="Filter to run cg analysis in [Default:r]")
    args = parser.parse_args()
    main(args)
