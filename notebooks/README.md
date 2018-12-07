Jupyter notebooks in this folder show the galaxies used in cg bias analysis as well as plot the results
A detailed explanation of measurement techniques and results can be found
[here](https://www.overleaf.com/read/wqztwvtnxhvn) (work in progress).

# CG_tests_and_results

## Tests
The initial portion shows results predicted from CG analysis and hence serve as additional unit tests
complimentary to those in tests.py.
It contains the following tests
1. Different HSM methods
1. Weight function
1. PSF size
1. PSF alpha  

### Different HSM methods
Our definition of m_cg implies that all bias arises from color gradients "only" is incorporated into m_cg,
making it independent of biases from shape measurement algorithms. For a constant weight size, the CG
bias computed from different shear estimation methods in galsim.hsm : REGAUSS, KSB, LINEAR, BJ are equal
as expected.

### Weight function
CG bias arises when the weight function "weighs" different parts of the galaxy surface brightness profile
differently. Thus the magnitude of m_cg is inversely proportional to the size of the weight function,
dissapearing when no weight function is applied.

### PSF size
Intuitively we expect that when the size of the PSF is small compared to the size of the galaxy, bias on
shear, and hence m_cg is small. Also when the PSF is large, gradients in the galaxy profile don't matter.
It is useful to remember while analyzing the plots that the HLR of bulge and disks are 0.17 amd 1.2 arcsecs
respectively.


### PSF alpha
Larger the wavelength dependence of the PSF, larger will be the bias from CG, with m_cg being 0 for an
achromatic PSF. For this study, the size of the PSF scales with wavelength by an exponent alpha. Plot of 
of m_cg vs alpha shows the expected dependence.


## Additional Analysis
The notebook also shows CG bias for the following conditions different from the reference galaxy used above.
1. Chromatic Atmospheric PSF
1. I band results
1. With emission lines

### Chromatic Atmosphere PSF
The main study used only the wavelength dependence of atmospheric seeing. We measure the effect of
including DCR (diffractive chromatic refraction) on m_cg. Effect DCR depends on the observing zenith
angle; for large zenith angle the bias grows significantly.

### I band Results
m_cg is computed for the reference galaxy when observed in the i band. CG bias is smaller in the i band.
We explore possible reasons for this, by shifting the r band to overlap with i band and compare the results
We conclude that the smaller bias is because of fewer spectral features being observed in the i band which has
almost the same width as r band (i band is slightly smaller but this does not affect the result).

### With emission lines
We add emission lines to the disk spectra of 50 catsim galaxies and measure the difference in cg bias
and conclude that it does not vary the bias significantly.

# CRG_test_results
Test galsim.ChromaticRealGalaxy on reference and catsim galaxies to verify that chromatic features
are reproduced. We test CRG with true and polynomial SED and in the presence of noise

# AEGIS_cg_results.ipynb
Results from CG bias analysis of galaxies in AEGIS catalog with galsim.ChromaticRealGalaxy

# show_gal
Shows galaxy with CG used in CG bias analysis. The notebook doesn't show CG bias computations,
but provides visuals of galaxies in the analysis.
