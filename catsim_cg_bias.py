"""Code to measure bias from cg for catsim galaxies with and without noise.
Script computes this in multiple ways
1) parametric galaxy: bulge+disk galaxy canvolved with LSST chromatic PSF
is drawn in LSST r band. The cg bias is measured in cg_fns_reloaded.py
2) CRG with polynomial SED: Galaxy is drawn in HST I and V bands, given as
input to CRG (along with HST PSF images in I and V band). The CRG is then
canvolved with LSST chromatic PSF is drawn in LSST r band. The cg bias is
measured in cg_fns_reloaded.py. The CRG produced uses the default polynomial
SEDs.
3) CRG with polynomial SED: Galaxy is drawn in HST I and V bands, given as
input to CRG (along with HST PSF images in I and V band). The CRG is then
canvolved with LSST chromatic PSFis drawn in LSST r band. The cg bias is
measured in cg_fns_reloaded.py. The CRG produced takes the true SEDS of the
bulge and disk as input.
The script also computes SNR of the galaxy in a 1 second HST image (V/I) and
full depth HST image(r).
"""
import galsim
import extinction
import os
import numpy as np
from astropy.table import Table, Column
import cg_functions as cg_fn
galsim.ChromaticConvolution.resize_effective_prof_cache(5)


def get_rand_gal(params):
    """Returns galxies with cg that satisfy certain selection cuts"""
    path = '/nfs/slac/g/ki/ki19/deuce/AEGIS/LSST_cat/'
    cat_name = path + 'OneDegSq.dat'
    cat = Table.read(cat_name, format='ascii.basic')
    # get galaxies with color gradients
    cond1 = (cat['sedname_disk'] != '-1') & (cat['sedname_bulge'] != '-1') & (cat['sedname_agn'] == '-1')
    # reddening law fails for high redshift.
    cond2 = cat['redshift'] < 1.2
    # match magnitude cut of AEGIS
    cond3 = cat['i_ab'] <= 25.3
    # Dont pick too large galaxies
    cond4 = (cat['DiskHalfLightRadius'] < 3) & (cat['BulgeHalfLightRadius'] < 3)
    # cond5 = (cat['DiskHalfLightRadius'] > 0.2) ^ (cat['BulgeHalfLightRadius'] > 0.2)
    q, =  np.where(cond1 & cond2 & cond3 & cond4)  # & cond5)
    indices = range(int(params.num) * int(params.size), (int(params.num) + 1) * int(params.size))
    new_cat = cat[q]
    return new_cat[indices], indices


def get_npix(cat, scale, psf_sig):
    """Returns the number of pixelsrequired to draw a catsim galaxy.
    The dimensions are set by the larger of disk, bulge and psf.
    @cat         catsim row with galaxy parameters
    @scale       size of the pixel when the galaxy is drawn
    @psf_sig     size of the psf 
    returns number of pixels in x and y axes. 
    """
    a_max = max(cat['a_b'], cat['a_d'], 2 * psf_sig)
    b_max = max(cat['b_b'], cat['b_d'], 2 * psf_sig)
    ellip = 1 - b_max / a_max
    theta = cat['pa_bulge'] * np.pi / 180.
    nx = 5 * a_max * (np.abs(np.sin(theta)) + (1 - ellip) * np.abs(np.cos(theta)))
    ny = 5 * a_max * (np.abs(np.cos(theta)) + (1 - ellip) * np.abs(np.sin(theta)))
    return int(nx / scale), int(ny / scale)


def get_catsim_SED(sed_name, redshift=0., a_v=None,
                   r_v=3.1, model='ccm89'):
    """Returns SED of a galaxy in CatSim catalog corrected for redshift and
    extinction.
    @param sed_file    Name of file with SED information.
    @flux_norm         Multiplicative scaling factor to apply to the SED.
    @bandpass          GalSim bandpass object which models the transmission fraction.
    @ redshift         Redshift of the galaxy.
    @a_v               Total V band extinction, in magnitudes.
    @r_v               Extinction R_V parameter.
    @model             Dust model name in the object's rest frame.
    """
    sed_path = '/nfs/slac/g/ki/ki19/deuce/AEGIS/LSST_cat/galaxySED/'
    full_sed_name = sed_path + sed_name + '.gz'
    SED = galsim.SED(full_sed_name, wave_type='nm',
                     flux_type='flambda').thin(rel_err=1e-4)
    SED = redden(SED, a_v, r_v, model=model)
    SED = SED.atRedshift(redshift)
    return SED


def get_gal(Args, cat, get_seds=False):
    """Returns parametric bulge+dik galaxy (galsim Chromatic object).
    @Args        galaxy parametrs (same for all catsim galaxies)
    @cat         catsim row with galaxy parameters (different for every galaxy)
    @get_seds    if True, returns bulge and disk SEDs (default:False).
    """
    norm_d = cat['fluxnorm_disk']  # * 30  * np.pi * (6.67 * 100 / 2.)**2
    Args.d_SED = get_catsim_SED(cat['sedname_disk'], redshift=cat['redshift'],
                                a_v=cat['av_d'], r_v=cat['rv_d'],
                                model='ccm89') * norm_d
    norm_b = cat['fluxnorm_bulge']  # * 30 * np.pi * (6.67 * 100 / 2.)**2
    Args.b_SED = get_catsim_SED(cat['sedname_bulge'], redshift=cat['redshift'],
                                a_v=cat['av_b'], r_v=cat['rv_b'],
                                model='ccm89') * norm_b
    #  composite SED
    seds = [Args.b_SED * (1 / norm_b), Args.d_SED * (1 / norm_d)]
    Args.c_SED = Args.b_SED + Args.d_SED
    PSF = cg_fn.get_PSF(Args)
    gal = cg_fn.get_gal_cg(Args)
    con = galsim.Convolve([gal, PSF])
    if get_seds:
        return gal, PSF, con, seds
    else:
        return gal, PSF, con


def redden(SED, a_v, r_v, model):
    """ Returns a new SED with the specified extinction applied.  Note that
    this will truncate the wavelength range to lie between 91 nm and 6000 nm
    where the extinction correction law is defined.
    Catsim SED has bluelimit 16nm, so make sure we don't use high z
    ## Do not use redshifts higher than 1.4 !!!!!!!!!
    """
    return SED *(lambda w: 1 if np.any(w < 125) else 1/extinction.reddening(w*10, a_v=a_v,
                                                                            r_v=r_v, model=model))


def a_b2re_e(a, b):
    """Converts  semi major and minor axis to ellipticity and half light
    radius"""
    re = (a * b)**0.5
    e = (a - b) / float(a + b)
    return re, e


def get_HST_Bandpass(band):
    """Returns a Bandpass object for the catalog.
    Using similar code from real.py in Galsim
    """
    # Currently, have bandpasses available for HST COSMOS, AEGIS, and CANDELS.
    # ACS zeropoints (AB magnitudes) from
    # http://www.stsci.edu/hst/acs/analysis/zeropoints/old_page/localZeropoints#tablestart
    # WFC3 zeropoints (AB magnitudes) from
    # http://www.stsci.edu/hst/wfc3/phot_zp_lbn
    bps = {'F275W': ('WFC3_uvis_F275W.dat', 24.1305),
           'F336W': ('WFC3_uvis_F336W.dat', 24.6682),
           'F435W': ('ACS_wfc_F435W.dat', 25.65777),
           'F606W': ('ACS_wfc_F606W.dat', 26.49113),
           'F775W': ('ACS_wfc_F775W.dat', 25.66504),
           'F814W': ('ACS_wfc_F814W.dat', 25.94333),
           'F850LP': ('ACS_wfc_F850LP.dat', 24.84245),
           'F105W': ('WFC3_ir_F105W.dat', 26.2687),
           'F125W': ('WFC3_ir_F125W.dat', 26.2303),
           'F160W': ('WFC3_ir_F160W.dat', 25.9463)
           }
    try:
        bp = bps[band.upper()]
    except KeyError:
        raise ValueError("Unknown bandpass {0}".format(band))
    fn = os.path.join(galsim.meta_data.share_dir, "bandpasses", bp[0])
    bandpass = galsim.Bandpass(fn, wave_type='nm', zeropoint=bp[1])
    return bandpass.thin(rel_err=1e-4)


def get_eff_psf(chr_PSF, sed, bands):
    """returns galsim interpolated image of the effective PSF, i.e image of
    the chromatic PSF in each band, for a given SED.
    """
    V, I = bands[0], bands[1]
    star = galsim.Gaussian(half_light_radius=1e-9) * sed
    con = galsim.Convolve(chr_PSF, star)
    PSF_img_V = con.drawImage(V, nx=40, ny=40, scale=0.03)
    PSF_img_I = con.drawImage(I, nx=40, ny=40, scale=0.03)
    PSF_V = galsim.InterpolatedImage(PSF_img_V, flux=1.)
    PSF_I = galsim.InterpolatedImage(PSF_img_I, flux=1.)
    return [PSF_V, PSF_I]


def get_CRG(cat, rng, row):
    """Create CRG for a given input parametrs form catsim.
    Bulge + Disk galaxy is created, convolved with HST PSF, drawn in HST V and
    I bands for 1 second exposure. Correlated noise (computed from AEGIS images)
    is added to each image. SNR in a gaussian elliptical aperture is computed.
    cgr1: The galaxy +psf images + noise correlation function is provided as
    input to CRG with default polynomial SEDs.
    crg2: same as crg1 except, the input galaxy images are padded with noise.
    This enables us to to draw the CRG image larger than the input image,
    and not have boundary edges.
    crg3: same as crg2 except the SEDS of bulge and disk are provided as input
    to CRG.
    @cat    catsim row containig catsim galaxy parametrs.
    @rng    random number generator.
    @row    astropy table to save measurents.
    """
    #  HST scale
    scale = 0.03
    area = 4.437 * 10000  # np.pi * (2.4 * 100 / 2.)**2
    v_exptime = 1  # 2260
    i_exptime = 1  # 2100
    psf_sig = 0.06
    nx, ny = get_npix(cat, scale, psf_sig)
    print "Number of HST pixels:", nx, ny
    b_r, b_g = a_b2re_e(cat['a_b'], cat['b_b'])
    d_r, d_g = a_b2re_e(cat['a_d'], cat['b_d'])
    b_s = galsim.Shear(g=b_g, beta=cat['pa_bulge'] * galsim.degrees)
    d_s = galsim.Shear(g=d_g, beta=cat['pa_disk'] * galsim.degrees)
    input_p = cg_fn.Eu_Args(scale=scale, redshift=cat['redshift'],
                            disk_n=cat['bulge_n'], bulge_n=cat['disk_n'],
                            disk_HLR=d_r, bulge_HLR=b_r,
                            bulge_e=[b_s.e1, b_s.e2],
                            disk_e=[d_s.e1, d_s.e2],
                            psf_sig_o=0.071, psf_w_o=806,
                            bulge_frac=0.5)
    input_p.T_flux = 2
    gal, PSF, con, seds = get_gal(input_p, cat, get_seds=True)
    # get bandpass
    V = get_HST_Bandpass('F606W')
    I = get_HST_Bandpass('F814W')
    c_sed = seds[0] + seds[1]
    temp_d = seds[1] * cat['fluxnorm_disk']
    temp_b = seds[0] * cat['fluxnorm_bulge']
    b_sed_mag = [temp_b.calculateMagnitude(V), temp_b.calculateMagnitude(I)]
    d_sed_mag = [temp_d.calculateMagnitude(V), temp_d.calculateMagnitude(I)]
    row['b_mag'] = b_sed_mag
    row['d_mag'] = d_sed_mag
    gal_im_v = con.drawImage(V, nx=nx, ny=ny, scale=scale,
                             area=area, exptime=v_exptime)
    gal_im_i = con.drawImage(I, nx=nx, ny=ny, scale=scale,
                             area=area, exptime=i_exptime)
    flux = np.array([gal_im_v.array.sum(),
                     gal_im_i.array.sum()])
    snr_num = np.array([np.sum(gal_im_v.array**2), np.sum(gal_im_i.array**2)])
    # correlated noise to add to image
    noise_v = galsim.getCOSMOSNoise(file_name='data/acs_V_unrot_sci_cf.fits',
                                    rng=rng)
    noise_i = galsim.getCOSMOSNoise(file_name='data/acs_I_unrot_sci_cf.fits',
                                    rng=rng)
    # Add noise
    gal_im_v.addNoise(noise_v)
    gal_im_i.addNoise(noise_i)
    var_v = noise_v.getVariance()
    var_i = noise_i.getVariance()
    var = np.array([var_v, var_i])
    # Compute sn_ellip_gauss
    try:
        res_v = galsim.hsm.FindAdaptiveMom(gal_im_v)
        res_i = galsim.hsm.FindAdaptiveMom(gal_im_i)
        aperture_noise_v = np.sqrt(var_v * 2 * np.pi * (res_v.moments_sigma**2))
        aperture_noise_i = np.sqrt(var_i * 2 * np.pi * (res_i.moments_sigma**2))
        sn_ellip_gauss_v = res_v.moments_amp / aperture_noise_v
        sn_ellip_gauss_i = res_i.moments_amp / aperture_noise_i
    except:
        sn_ellip_gauss_v = -10
        sn_ellip_gauss_i = -10
    row['HST_sn_ellip_gauss'] = np.array([sn_ellip_gauss_v, sn_ellip_gauss_i])
    row['HST_noise_var'] = var
    row['HST_flux'] = flux
    row['HST_SNR'] = np.sqrt(snr_num / var)
    # covariance function for CRG input
    xi_v = galsim.getCOSMOSNoise(file_name='data/acs_V_unrot_sci_cf.fits',
                                 variance=var_v, rng=rng)
    xi_i = galsim.getCOSMOSNoise(file_name='data/acs_I_unrot_sci_cf.fits',
                                 variance=var_i, rng=rng)
    eff_PSFs = get_eff_psf(PSF, c_sed, [V, I])
    print "Creating CRG with noise padding"
    cg_size = int(max(cat['BulgeHalfLightRadius'],
                      cat['DiskHalfLightRadius'], 2 * psf_sig) * 12)
    print "pad size", cg_size
    intp_gal_v = galsim.InterpolatedImage(gal_im_v, noise_pad=noise_v,
                                          noise_pad_size=cg_size)
    gal_im_v_pad = intp_gal_v._pad_image
    intp_gal_i = galsim.InterpolatedImage(gal_im_i, noise_pad=noise_i,
                                          noise_pad_size=cg_size)
    gal_im_i_pad = intp_gal_i._pad_image
    print "CRG input im shape ", gal_im_v_pad.array.shape[0] * scale
    #  Polynomial SEDs
    crg1 = galsim.ChromaticRealGalaxy.makeFromImages(images=[gal_im_v_pad, gal_im_i_pad],
                                                     bands=[V, I],
                                                     xis=[xi_v, xi_i],
                                                     PSFs=eff_PSFs)
    #  True SEDs
    crg2 = galsim.ChromaticRealGalaxy.makeFromImages(images=[gal_im_v_pad, gal_im_i_pad],
                                                     bands=[V, I],
                                                     xis=[xi_v, xi_i],
                                                     PSFs=eff_PSFs,
                                                     SEDs=seds)
    return crg1, crg2


def get_flux(ab_magnitude, exposure_time, zero_point):
    """Convert source magnitude to flux.
    The calculation includes the effects of atmospheric extinction.
    Args:
        ab_magnitude(float): AB magnitude of source.
    Returns:
        float: Flux in detected electrons.
    airmass =1.2
    zeropint is at airmass 1.2
    """
    return exposure_time*zero_point*10**(-0.4*(ab_magnitude - 24))


def get_lsst_noise(cat, rng, row):
    """Create CRG for a given input parametrs form catsim.
    Args:
    cat: astropy row containig catsim galaxy parametrs
    """
    #  HST scale
    exposure_time = 5520  # 6900
    r_zero_point = 43.70  # 44.80
    i_zero_point = 32.36  # 33.17
    area = 32.4 * 10000  # np.pi * (667 / 2.)**2
    scale = 0.2
    r_sky_brightness = 21.2
    i_sky_brightness = 20.5
    #  r0 = 6.67 * 100 / 2.
    psf_sig = 0.297
    npix = int(max(cat['BulgeHalfLightRadius'],
                   cat['DiskHalfLightRadius'], 2 * psf_sig) * 8 / scale)
    print "Number of pixels is ", npix
    b_r, b_g = a_b2re_e(cat['a_b'], cat['b_b'])
    d_r, d_g = a_b2re_e(cat['a_d'], cat['b_d'])
    b_s = galsim.Shear(g=b_g, beta=cat['pa_bulge'] * galsim.degrees)
    d_s = galsim.Shear(g=d_g, beta=cat['pa_disk'] * galsim.degrees)
    input_p = cg_fn.LSST_Args(scale=scale, npix=npix,
                              redshift=cat['redshift'], bulge_frac=0.5,
                              disk_n=cat['bulge_n'], bulge_n=cat['disk_n'],
                              disk_HLR=d_r, bulge_HLR=b_r,
                              bulge_e=[b_s.e1, b_s.e2],
                              disk_e=[d_s.e1, d_s.e2])
    r = galsim.Bandpass('data/baseline/total_r.dat',
                        wave_type='nm').thin(rel_err=1e-4)
    input_p.bp = r
    input_p.T_flux = 2
    gal, PSF, con = get_gal(input_p, cat)
    r = galsim.Bandpass('data/baseline/total_r.dat', zeropoint='AB',
                        wave_type='nm').thin(rel_err=1e-4)
    i = galsim.Bandpass('data/baseline/total_i.dat', zeropoint='AB',
                        wave_type='nm').thin(rel_err=1e-4)
    # old throughputs did not have one of the mirrors
    # get bandpass
    # r = galsim.Bandpass('data/LSST_r.dat', zeropoint='AB',
    #                     wave_type='nm').thin(rel_err=1e-4)
    # i = galsim.Bandpass('data/LSST_i.dat', zeropoint='AB',
    #                    wave_type='nm').thin(rel_err=1e-4)

    # draw galaxy image
    gal_im_r = con.drawImage(r, nx=npix, ny=npix, scale=scale, area=area,
                             exptime=exposure_time)
    gal_im_i = con.drawImage(i, nx=npix, ny=npix, scale=scale, area=area,
                             exptime=exposure_time)
    flux = np.array([gal_im_r.array.sum(),
                     gal_im_i.array.sum()])
    snr_num = np.array([np.sum(gal_im_r.array**2), np.sum(gal_im_i.array**2)])
    # correlated noise to add to image
    r_mean_sky_level = get_flux(r_sky_brightness, exposure_time,
                                r_zero_point) * scale**2
    i_mean_sky_level = get_flux(i_sky_brightness, exposure_time,
                                i_zero_point) * scale**2
    noise_r = galsim.PoissonNoise(rng=rng, sky_level=r_mean_sky_level)
    noise_i = galsim.PoissonNoise(rng=rng, sky_level=i_mean_sky_level)
    # Add noise
    gal_im_r.addNoise(noise_r)
    gal_im_i.addNoise(noise_i)
    var_r = noise_r.getVariance()
    var_i = noise_i.getVariance()
    var = np.array([var_r, var_i])
    # Compute sn_ellip_gauss
    try:
        res_r = galsim.hsm.FindAdaptiveMom(gal_im_r)
        res_i = galsim.hsm.FindAdaptiveMom(gal_im_i)
        aperture_noise_r = np.sqrt(var_r * 2 * np.pi * (res_r.moments_sigma**2))
        aperture_noise_i = np.sqrt(var_i * 2 * np.pi * (res_i.moments_sigma**2))
        sn_ellip_gauss_r = res_r.moments_amp / aperture_noise_r
        sn_ellip_gauss_i = res_i.moments_amp / aperture_noise_i
        row['mom_sigma'] = [res_r.moments_sigma, res_i.moments_sigma]
    except:
        sn_ellip_gauss_r = -10
        sn_ellip_gauss_i = -10
    row['LSST_sn_ellip_gauss'] = np.array([sn_ellip_gauss_r, sn_ellip_gauss_i])
    row['LSST_noise_var'] = var
    row['LSST_flux'] = flux
    print "LSST flux", flux
    print "LSST noise var ", var
    row['LSST_SNR'] = np.sqrt(snr_num / var)
    t_flux_r = get_flux(cat['r_ab'], exposure_time, r_zero_point)
    t_flux_i = get_flux(cat['i_ab'], exposure_time, i_zero_point)
    t_fluxs = np.array([t_flux_r, t_flux_i])
    row['LSST_target_flux'] = t_fluxs


def meas_cg_bias(gal, row, f_name,
                 rt_g, f_type, npix=360):
    """Computes bias due to color gradient on sahpe measuremnt.
    For an input chromatic galaxy with cg,  gal an equilvalent galaxy with
    no cg is created and the shear recovered from each (with ring test) is
    measured.
    @gal     input galaxy with cg.
    @row     astropy table row to save measured shear.
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


def get_lsst_para(cat):
    """Create parametric bulge+disk galaxy for input parametrs form catsim.
    @cat: catsim row containig catsim galaxy parametrs
    """
    #  HST scale
    scale = 0.2
    b_r, b_g = a_b2re_e(cat['a_b'], cat['b_b'])
    d_r, d_g = a_b2re_e(cat['a_d'], cat['b_d'])
    b_s = galsim.Shear(g=b_g, beta=cat['pa_bulge'] * galsim.degrees)
    d_s = galsim.Shear(g=d_g, beta=cat['pa_disk'] * galsim.degrees)
    input_p = cg_fn.LSST_Args(scale=scale, redshift=cat['redshift'],
                              bulge_n=cat['disk_n'], disk_n=cat['bulge_n'],
                              disk_HLR=d_r, bulge_HLR=b_r,
                              bulge_e=[b_s.e1, b_s.e2],
                              disk_e=[d_s.e1, d_s.e2],
                              bulge_frac=0.5)
    input_p.T_flux = 2
    gal, PSF, con = get_gal(input_p, cat)
    return gal


def main(params):
    """Measures bias from cg for a parametric galaxy and the same galaxy through CRG """
    print 'Running on num {0}'.format(params.num)
    # input shear
    g = np.linspace(0.005, 0.01, 2)
    rt_g = np.array([g, g]).T
    # Number of catsim galaxies to run
    num = int(params.size)
    catsim_gals, indexs = get_rand_gal(params)
    rng = galsim.BaseDeviate(int(params.num))
    #  Create Table to save result
    names = ('NUMBER', 'C_galtileid', 'redshift')
    dtype = ('int', 'int', 'float')
    cols = [indexs, catsim_gals['galtileid'], catsim_gals['redshift']]
    index_table = Table(cols, names=names, dtype=dtype)
    col = Column(np.zeros([num, len(rt_g), 2]), name='rt_g',
                 shape=(2, 2), dtype='f8')
    index_table.add_column(col)
    col = Column(np.zeros([num, len(rt_g), 2]), name='para_g_cg',
                 shape=(2, 2), dtype='f8')
    index_table.add_column(col)
    col = Column(np.zeros([num, len(rt_g), 2]), name='para_g_no_cg',
                 shape=(2, 2), dtype='f8')
    index_table.add_column(col)
    col = Column(np.zeros([num, len(rt_g), 2]), name='CRG_g_cg',
                 shape=(2, 2), dtype='f8')
    index_table.add_column(col)
    col = Column(np.zeros([num, len(rt_g), 2]), name='CRG_g_no_cg',
                 shape=(2, 2), dtype='f8')
    index_table.add_column(col)
    col = Column(np.zeros([num, len(rt_g), 2]), name='CRG_tru_sed_g_cg',
                 shape=(2, 2), dtype='f8')
    index_table.add_column(col)
    col = Column(np.zeros([num, len(rt_g), 2]), name='CRG_tru_sed_g_no_cg',
                 shape=(2, 2), dtype='f8')
    index_table.add_column(col)
    col = Column(np.zeros([num, 2]), name='b_mag',
                 shape=(2, ), dtype='f8')
    index_table.add_column(col)
    col = Column(np.zeros([num, 2]), name='d_mag',
                 shape=(2, ), dtype='f8')
    index_table.add_column(col)
    col = Column(np.zeros([num, 2]), name='HST_sn_ellip_gauss',
                 shape=(2, ), dtype='f8')
    index_table.add_column(col)
    col = Column(np.zeros([num, 2]), name='HST_SNR',
                 shape=(2, ), dtype='f8')
    index_table.add_column(col)
    col = Column(np.zeros([num, 2]), name='HST_noise_var',
                 shape=(2, ), dtype='f8')
    index_table.add_column(col)
    col = Column(np.zeros([num, 2]), name='HST_flux',
                 shape=(2, ), dtype='f8')
    index_table.add_column(col)
    col = Column(np.ones([num, 2]) * -1, name='mom_sigma',
                 shape=(2, ), dtype='f8')
    index_table.add_column(col)
    col = Column(np.zeros([num, 2]), name='LSST_sn_ellip_gauss',
                 shape=(2, ), dtype='f8')
    index_table.add_column(col)
    col = Column(np.zeros([num, 2]), name='LSST_SNR',
                 shape=(2, ), dtype='f8')
    index_table.add_column(col)
    col = Column(np.zeros([num, 2]), name='LSST_noise_var',
                 shape=(2, ), dtype='f8')
    index_table.add_column(col)
    col = Column(np.zeros([num, 2]), name='LSST_flux',
                 shape=(2, ), dtype='f8')
    index_table.add_column(col)
    col = Column(np.zeros([num, 2]), name='LSST_target_flux',
                 shape=(2, ), dtype='f8')
    index_table.add_column(col)
    for n in range(num):
        scale = 0.2
        psf_sig = 0.297
        z = catsim_gals[n]['redshift']
        print "Creating gal number {0} at redshift {1} ".format(n, z)
        npix = int(max(catsim_gals[n]['BulgeHalfLightRadius'],
                       catsim_gals[n]['DiskHalfLightRadius'],
                       2 * psf_sig) * 8 / scale)
        print "number of LSST pixels", npix
        CRG1, CRG2 = get_CRG(catsim_gals[n], rng, index_table[n])
        print "measuring cg bias for CRG"
        meas_cg_bias(CRG1, index_table[n], params.filter,
                     rt_g, 'CRG', int(npix))
        meas_cg_bias(CRG2, index_table[n], params.filter,
                     rt_g, 'CRG_tru_sed', int(npix))
        #  CG bias for lsst with no noise
        npix = int(max(catsim_gals[n]['BulgeHalfLightRadius'],
                       catsim_gals[n]['DiskHalfLightRadius'],
                       2 * psf_sig) * 30 / scale)
        para_gal = get_lsst_para(catsim_gals[n])
        meas_cg_bias(para_gal, index_table[n], params.filter,
                     rt_g, 'para', npix)
        get_lsst_noise(catsim_gals[n], rng, index_table[n])
        index_table['rt_g'] = rt_g
    path = '/nfs/slac/g/ki/ki19/deuce/AEGIS/data_test_CRG/results/cg_test/full/CRG_catsim_padded/'
    op_file = path + 'catsim_cg_bias_{0}_{1}_band.fits'.format(params.num,
                                                               params.filter)
    index_table.write(op_file, format='fits',
                      overwrite=True)
    print "Saving output at", op_file


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--size', default=10,
                        help="Number of entries [Default:10]")
    parser.add_argument('--num', default='0',
                        help="File number [Default:0]")
    parser.add_argument('--filter', default='r',
                        help="Filter to run cg analysis in [Default:r]")
    args = parser.parse_args()
    main(args)
