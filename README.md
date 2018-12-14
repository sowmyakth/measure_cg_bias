[![Code Health](https://landscape.io/github/sowmyakth/measure_cg_bias/master/landscape.svg?style=flat)](https://landscape.io/github/sowmyakth/measure_cg_bias/master)
# measure_cg_bias
Code to measure bias from galaxy color gradients in weak lensing measurements.
CG bias is estimated for
1. reference parametric galaxy with bulge and disk
2. Galaxies from CatSim
3. HST V/I band images from AEGIS


/notebooks directory contains jupyter notebooks containing plots showing results
from analysis and can be used to visulaize the galaxies and PSFs used in the
analysis.

/data directory contains template SEDs, LSST filter response curves, and HST noise
correlation functions used in the analysis.

Overleaf file (work in progress) describing measurement techniques and results can be found
[here](https://www.overleaf.com/read/wqztwvtnxhvn).

## Dataset
1. The AEGIS [catalog](http://great3.jb.man.ac.uk/leaderboard/data/public/AEGIS_training_sample.tar.gz)
with H/I postage stamp images of isolated galaxies and phometric measuremnts. Detailed
doument describing how the catalog was created and can be found [here](https://docs.google.com/viewer?url=https://github.com/sowmyakth/measure_cg_bias/raw/master/pdfs/Reducing_AEGIS_gal.pdf)
2. CatSim

