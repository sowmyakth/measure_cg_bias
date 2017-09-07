Jupyter notebboks in this folder show the galaxies used in cg bias analysis as well as plot the results
A detailed explanation of measurement techniques and results can be found
[here](https://www.overleaf.com/read/wqztwvtnxhvn) (work in progress).

# CG_tests_and_results

## Tests
The initial portion shows results expected from CG analysis and hence serve as additional unit tests
complimentary tothose in tests.py.
It contains the following tests
1. Different HSM methods
1. Weight function
1. PSF size
1. PSF alpha  

### Different HSM methods
Our definition of m_cg implies that all bias arises from color gradients only is incorporated into m_cg,
making it independent of biases from shape measurement algorithms. For a constant weight size, the CG 
bias computed from different shear estimation methods in galsim.hsm : REGAUSS, KSB, LINEAR, BJ are equal
as expected. 

### Weight function
CG bias arises when the weight function "weighs" differnt parts of the galaxy surface brightness profile
differently. Thus the magnitude og m_cg is inversely proportional to the size of the weight function, 
dissapering when no weight function is applied.

### PSF size
Intutively we expect that when the size of the PSF is small compared to the size of the galaxy bias on 
shear, and hence m_cg is small. Also when the PSF is large, gradients in the galaxy profile dont matter.
It is useful to remember while analyzing the plots that the HLR of bulge and disks are 0.17 amd 1.2 arcsecs
respectively.


### PSF alpha
Larger the wavlength dependence of the PSF, larger will be the bias from CG, with m_cg being 0 for an
achromatic PSF. For this study the size of the PSF scales with wavlength by an exponent alpha. Plot of 
of m_cg vs alpha shows the expected dependence.


## Additional Analysis
The notebook also shows CG bias for the following conditions differnt from the reference galaxy used above.
1. Chromatic Atmosphere PSF
1. I band results
1. With emission lines

### Chromatic Atmosphere PSF
The main study used only the wavelength dependence of atmospheric seeing. We measure the effect of
including DCR (diffractive chromatic refraction) on m_cg. Effect DCR depends on teh observing zenith
angle; for large zenith angle the bias grows significantly.

### I band Results
m_cg is computed for the reference galaxy when observed in the i band. CG bias is smaller in th i band.
We explore possible reasons for this, by shifting the r band to overlap with i band and compare the results
We conclude that the smaller bias is because of fewer spectral features being observed in the i band which has
almost the same width as r band (i band is slightly smaller but this does not effect the result).

### With emmision lines
We add emission lines to the disk spectra of 50 catsim galaxies and measure the difference in cg bias
and conclude that is does varu the bias significantly.

# CRG_test_results
Test galsim.ChromaticRealGalaxy on reference and catsim galaxies to verify that chromatic features
are reproduced. we test CRG with true and polynomial SED and in the presence of noise

# AEGIS_cg_results.ipynb
Results from CG bias analysis of galaxies in AEGIS catalog with galsim.ChromaticRealGalaxy

# show_gal
Shows galaxy with CG used in CG bias analysis. The notebook doesn't show CG bias computations,
but provides visuals of galaxies in the analysis.
