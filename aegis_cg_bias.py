"""This code computes bias from cg expected from galaxies from AEGIS catalog
when observed using LSST. The AEGIS galaxy images in v/I bands are provided
as input to ChromaticRealGalaxy module of galasim to create surface brightness
profile of the galaxy which is used to measure the bias from color gradient
when seen with LSST. The code reads the real galaxy catalog which uses
RealGalaxy object in V/I bands to produce CRG. The result is saved along with
the corresponding entries in the complete catalog.
"""
import galsim
import numpy as np
from astropy.table import Table, Column, hstack
import cg_functions as cg_fn
galsim.ChromaticConvolution.resize_effective_prof_cache(5)


def get_xis(rgc, use_index, rng):
    """Returns correlation function for a given entry in RealGalaxy Catalog"""
    xis = []
    for r in rgc:
        noise_image, pixel_scale, var = r.getNoiseProperties(use_index)
        # Make sure xi image is odd-sized.
        if noise_image.array.shape[0] % 2 == 0:
            bds = noise_image.bounds
            new_bds = galsim.BoundsI(bds.xmin + 1,
                                     bds.xmax, bds.ymin + 1,
                                     bds.ymax)
            noise_image = noise_image[new_bds]
        ii = galsim.InterpolatedImage(noise_image, normalization='sb',
                                      calculate_stepk=False,
                                      calculate_maxk=False,
                                      x_interpolant='linear')
        xi = galsim.correlatednoise._BaseCorrelatedNoise(rng, ii,
                                                         noise_image.wcs)
        xi = xi.withVariance(var)
        xis.append(xi)
    return xis


def get_CRG(rgc, index, rng,
            noise_pad=False, noise_pad_size=0):
    """Create CRG for a given input parametrs form RealGalaxyCatalog rgc.
    PSF convolved galaxy image and psf image is obtained from rgc for v/I.
    The input galaxy images are padded with noise. This enables us to to draw
    the CRG image larger than the input image, and not have boundary edges.
    @rgc               RealGalaxy catalog with galaxy to run analysis on.
    @index             index of galaxy in rgc
    @rng               random number generator.
    @noise_pad         If True pad image with noise.
    @noise_pad_size    Size in arcseconds to which te image is padded
    returns ChromatocRealGalaxy computed
    """
    bands = [rgc[0].getBandpass(), rgc[1].getBandpass()]
    im1 = rgc[0].getGalImage(index)
    print "RG pstamp size", im1.array.shape
    noise1 = rgc[0].getNoise(index)
    im2 = rgc[1].getGalImage(index)
    noise2 = rgc[1].getNoise(index)
    print noise_pad
    if noise_pad is True:
        print "Adding noise pad of size ", noise_pad_size
        v_intp = galsim.InterpolatedImage(im1, noise_pad=noise1,
                                          noise_pad_size=noise_pad_size)
        gal_im_v = v_intp._pad_image
        i_intp = galsim.InterpolatedImage(im2, noise_pad=noise2,
                                          noise_pad_size=noise_pad_size)
        gal_im_i = i_intp._pad_image
        imgs = [gal_im_v, gal_im_i]
    else:
        imgs = [im1, im2]
    PSFs = [rgc[0].getPSF(index), rgc[1].getPSF(index)]
    xis = get_xis(rgc, index, rng)
    CRG = galsim.ChromaticRealGalaxy.makeFromImages(images=imgs,
                                                    bands=bands,
                                                    xis=xis,
                                                    PSFs=PSFs)
    return CRG


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
    @npix    Number of pixels to draw image for cg measurement.
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


def main(params):
    """Creates CRG and compute cg bias. Input galaxy and noise added
    is sampled from AEGIS input catalog"""
    print 'Running on num {0}'.format(params.num)
    # input shear
    g = np.linspace(0.005, 0.01, 2)
    rt_g = np.array([g, g]).T
    # Number of catsim galaxies to run
    num = int(params.size)
    indexs = range(int(params.num) * num, (int(params.num) + 1) * num)
    # indexs = range(26500, 26517)
    in_filters = ['f606w', 'f814w']
    file_filter_name = ['V', 'I']
    rgc_path = '/nfs/slac/g/ki/ki19/deuce/AEGIS/AEGIS_catalog_full/AEGIS_training_sample/'
    rgc_cat_name = 'AEGIS_galaxy_catalog_filter_25.2.fits'
    comp_path = '/nfs/slac/g/ki/ki19/deuce/AEGIS/AEGIS_catalog_full/'
    comp_cat_name = 'complete_AEGIS_galaxy_catalog_filter_25.2.fits'
    rgc, comp_cat = [], {}
    for f, filt in enumerate(in_filters):
        name = rgc_cat_name.replace('filter', file_filter_name[f])
        rgc.append(galsim.RealGalaxyCatalog(name, dir=rgc_path))
        name = comp_cat_name.replace('filter', file_filter_name[f])
        comp_cat[filt] = Table.read(comp_path + name, format='fits')[indexs]
    #  Create Table to save result
    index_table = hstack([comp_cat['f606w'], comp_cat['f814w']])
    col = Column(np.zeros([num, len(rt_g), 2]), name='rt_g',
                 shape=(2, 2), dtype='f8')
    index_table.add_column(col)
    col = Column(np.ones([num, len(rt_g), 2]) * -1, name='CRG_g_cg',
                 shape=(2, 2), dtype='f8')
    index_table.add_column(col)
    col = Column(np.ones([num, len(rt_g), 2]) * -1, name='CRG_g_no_cg',
                 shape=(2, 2), dtype='f8')
    index_table.add_column(col)
    large_gal_index = np.array([1523, 14529, 15285, 17531, 18393, 19506])
    for n in range(num):
        indx = indexs[n]
        print "Running on index ", indx
        rng = galsim.random.BaseDeviate(indx)
        assert((comp_cat[filt]['IDENT'][n] == index_table['IDENT_1'][n]))
        assert((index_table['IDENT_1'][n] == index_table['IDENT_2'][n]))
        rg_v = galsim.RealGalaxy(rgc[0], id=index_table['IDENT_1'][n])
        assert(indx == rg_v.index)
        index_table['rt_g'][n] = rt_g
        if not np.all(large_gal_index - indx):
            print "Large galaxy; skipping computation"
            continue
        #  ident = index_table['IDENT_1'][n]
        psf_sig = 0.297
        npix = int(max(index_table['FLUX_RADIUS_1'][n] * 0.03,
                       index_table['FLUX_RADIUS_2'][n] * 0.03,
                       2 * psf_sig) * 8 / 0.2)
        hlrs = [index_table['FLUX_RADIUS_1'][n] * 0.03,
                index_table['FLUX_RADIUS_2'][n] * 0.03]
        print "HST HLR ", hlrs
        print "Number of pixels for cg measurement ", npix
        noise_pad_size = (npix * 0.2 * 1.5)
        CRG = get_CRG(rgc, indx, rng,
                      noise_pad=True, noise_pad_size=noise_pad_size)
        meas_cg_bias(CRG, index_table[n], params.filter,
                     rt_g, 'CRG', npix)
    # path = '/nfs/slac/g/ki/ki19/deuce/AEGIS/data_test_CRG/results/cg_test/full/CRG_AEGIS/'
    path = 'delete/'
    op_file = path + 'AEGIS_cg_CRG_{0}_{1}_band.fits'.format(params.num,
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
