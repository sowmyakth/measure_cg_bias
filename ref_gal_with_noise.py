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
from astropy.table import Table, Column
import cg_functions as cg_fn
galsim.ChromaticConvolution.resize_effective_prof_cache(5)


def get_rand_SNR(size):
    filters = ['f606w', 'f814w']
    file_filter_name = ['V', 'I']
    path = '/nfs/slac/g/ki/ki19/deuce/AEGIS/AEGIS_catalog_full/'
    name = 'complete_AEGIS_galaxy_catalog_filter_25.2.fits'
    selec_cat = {}
    for f in range(len(filters)):
        selec_name = name.replace('filter', file_filter_name[f])
        selec_cat[f] = Table.read(path + selec_name, format='fits')
    indices = np.random.randint(len(selec_cat[0]), size=size)
    V_SNR = selec_cat[0]['sn_ellip_gauss'][indices]
    I_SNR = selec_cat[1]['sn_ellip_gauss'][indices]
    SNRs = np.array([V_SNR, I_SNR]).T
    return SNRs


def get_table(num):
    names = ('NUMBER', 'redshift', 'sn_ellip_gauss', 'input_SNR', 'noise_var',
             'flux', 'CRG_g_cg', 'CRG_g_no_cg', 'CRG_m1', 'CRG_m2', 'CRG_c1',
             'CRG_c2')
    dtype = ['int'] + ['float'] * 11
    cols = [range(num), np.zeros(num)] + [np.ones([num, 2]) * -10] * 4
    cols += [np.ones([num, 2, 2]) * -10] * 2 + [np.ones([num, 3]) * -10] * 4
    index_table = Table(cols, names=names, dtype=dtype)
    return index_table


def get_HST_im(con):
    scale = 0.03
    area = 4.437 * 10000  # np.pi * (2.4 * 100 / 2.)**2
    v_exptime = 1  # 2260
    i_exptime = 1  # 2100
    nx, ny = 360, 360   # number of x and y pixels
    # get bandpass
    V_band = cg_fn.get_HST_Bandpass('F606W')
    I_band = cg_fn.get_HST_Bandpass('F814W')
    # draw image
    gal_im_v = con.drawImage(V_band, nx=nx, ny=ny, scale=scale,
                             area=area, exptime=v_exptime)
    gal_im_i = con.drawImage(I_band, nx=nx, ny=ny, scale=scale,
                             area=area, exptime=i_exptime)
    return [gal_im_v, gal_im_i], [V_band, I_band]


def get_HST_im_noise(images, SNRs, row, rng):
    [gal_im_v, gal_im_i] = images
    flux = np.array([gal_im_v.array.sum(),
                     gal_im_i.array.sum()])
    xi_v = galsim.getCOSMOSNoise(file_name='data/acs_V_unrot_sci_cf.fits',
                                 rng=rng)
    xi_i = galsim.getCOSMOSNoise(file_name='data/acs_I_unrot_sci_cf.fits',
                                 rng=rng)
    var_v = gal_im_v.addNoiseSNR(xi_v, snr=SNRs[0],
                                 preserve_flux=True)
    var_i = gal_im_i.addNoiseSNR(xi_i, snr=SNRs[1],
                                 preserve_flux=True)
    xi_v = galsim.getCOSMOSNoise(file_name='data/acs_V_unrot_sci_cf.fits',
                                 variance=var_v, rng=rng)
    xi_i = galsim.getCOSMOSNoise(file_name='data/acs_I_unrot_sci_cf.fits',
                                 variance=var_i, rng=rng)
    # Compute sn_ellip_gauss
    res_v = galsim.hsm.FindAdaptiveMom(gal_im_v, strict=False)
    res_i = galsim.hsm.FindAdaptiveMom(gal_im_i, strict=False)
    if (res_v.error_message != "") or (res_v.error_message != ""):
        print "HSM failed"
    else:
        aperture_noise_v = np.sqrt(var_v * 2 * np.pi * (res_v.moments_sigma**2)) / 1
        aperture_noise_i = np.sqrt(var_i * 2 * np.pi * (res_i.moments_sigma**2)) / 1
        sn_ellip_gauss_v = res_v.moments_amp / aperture_noise_v
        sn_ellip_gauss_i = res_i.moments_amp / aperture_noise_i
        row['sn_ellip_gauss'] = np.array([sn_ellip_gauss_v, sn_ellip_gauss_i])
    row['noise_var'] = np.array([var_v, var_i])
    row['flux'] = flux
    row['input_SNR'] = SNRs
    return [gal_im_v, gal_im_i], [xi_v, xi_i]


def get_CRG(in_p, SNRs, row, rng):
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
    in_p.b_SED, in_p.d_SED, in_p.c_SED = cg_fn.get_template_seds(in_p)
    PSF = cg_fn.get_gaussian_PSF(in_p)
    gal = cg_fn.get_gal_cg(in_p)
    con = galsim.Convolve([gal, PSF])
    images, bands = get_HST_im(con)
    psf_v = cg_fn.get_eff_psf(PSF, in_p.c_SED, bands[0])
    psf_i = cg_fn.get_eff_psf(PSF, in_p.c_SED, bands[1])
    eff_PSFs = [psf_v, psf_i]
    images, cfs = get_HST_im_noise(images, SNRs, row, rng)
    crg1 = galsim.ChromaticRealGalaxy.makeFromImages(images=images,
                                                     bands=bands,
                                                     xis=cfs,
                                                     PSFs=eff_PSFs)
    return crg1


def get_lsst_para(in_p):
    """Create parametric bulge+disk galaxy for input parametrs."""
    in_p.b_SED, in_p.d_SED, in_p.c_SED = cg_fn.get_template_seds(in_p)
    gal = cg_fn.get_gal_cg(in_p)
    return gal


def meas_cg_bias(gal, row, meas_args,
                 psf_args, f_type):
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
    gcg, gnocg = cg_fn.calc_cg_crg(gal, meas_args, psf_args)
    print "Computing CG bias"
    if (gcg == "Fail") or (gnocg == "Fail"):
        print "HSM FAILED"
        return
    row[f_type + '_g_cg'] = gcg.T
    row[f_type + '_g_no_cg'] = gnocg.T
    gtrue = meas_args.rt_g
    m, c = cg_fn.get_bias(gcg[0], gnocg[0], gtrue.T[0])
    row[f_type + '_m1'] = m
    row[f_type + '_c1'] = c
    m, c = cg_fn.get_bias(gcg[1], gnocg[1], gtrue.T[1])
    row[f_type + '_m2'] = m
    row[f_type + '_c2'] = c


def main(Args):
    """Creates galaxy and psf images with different parametrs
     and measures cg bias. Varying LSST  PSF size"""
    # Set disk SED name
    e_s = [0.3, 0.3]
    filt = Args.filter
    g = np.linspace(0.005, 0.01, 2)
    rt_g = np.array([g, g]).T
    npix = 360
    SNRs = get_rand_SNR(int(Args.size))
    num = int(Args.size)
    rng = galsim.BaseDeviate(int(Args.file_num))
    index_table = get_table(num)
    indexs = range(int(Args.file_num) * num, (int(Args.file_num) + 1) * num)
    col = Column(np.ones(num) * -10, name='psf_sigma')
    index_table.add_column(col)
    dSED = Args.disk_SED_name
    for n in range(num):
        print "Computing for noise realization {0} in {1} band".format(n,
                                                                       filt)
        index_table['NUMBER'][n] = indexs[n]
        input_p = cg_fn.Eu_Args(scale=0.03, disk_SED_name=dSED,
                                bulge_e=e_s, disk_e=e_s,
                                psf_sig_o=0.071, psf_w_o=806)
        index_table['redshift'][n] = input_p.redshift
        CRG1 = get_CRG(input_p, SNRs[n], index_table[n], rng)
        # Compute CG bias
        meas_args = cg_fn.meas_args(rt_g=rt_g, npix=npix)
        meas_args.bp = galsim.Bandpass('data/baseline/total_%s.dat'%filt,
                                       wave_type='nm').thin(rel_err=1e-4)
        psf_args = cg_fn.psf_params()
        meas_cg_bias(CRG1, index_table[n], meas_args,
                     psf_args, 'CRG')
    path = 'results/'
    # path = "/nfs/slac/g/ki/ki19/deuce/AEGIS/cg_results/ref_gal_results/with_aeg_noise_all/"
    op_file = path + 'ref_gal_cg_bias_{0}_band_var_noise_{1}.fits'.format(filt,
                                                                          Args.file_num)
    index_table.write(op_file, format='fits',
                      overwrite=True)
    print "Saving output at", op_file


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--disk_SED_name', default='Im',
                        help="disk SED, one of E, Im, Sbc, Scd.[Default:Im]")
    parser.add_argument('--filter', default='r',
                        help="Filter to run cg analysis in [Default:r]")
    parser.add_argument('--size', default='10',
                        help="Number of noise realizations [Default:10]")
    parser.add_argument('--file_num', default='10',
                        help="Number to save the output file [Default:0]")
    args = parser.parse_args()
    main(args)
