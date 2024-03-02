# Code developed and maintained exclusively by Eitan Rapaport, Ritoban Basu Thakur
# Spectro-Imaging-Forecasting-Tool (SIFT)
Tool for forecasting Spectrometric and Photometric devices in the study of Sunyaev-Zeldovich distortions.
Includes CIB contamination with SIDES catalog, and CMB anisotropies as contaminants.
Resolves for y-value, electron temperature, peculiar velocity, profile of CIB, CMB anisotropy in an MCMC.

Requires SZpack for use.
Also requires Pandas, pickle, AstroPy, CAMB, NumPy, Matplotlib, Corner, Emcee, SciPy, multiprocessing, pygtc, matplotlib

# Requires particular files:

sides.csv:

A file with the averaged distribution of all spectra from continuum.fits, from SIDES

continuum.fits:

Spectra of CIB from SIDES catalog

Mather_photonNEP12a.py:

File to calculate photon NEP.

# TO RUN:

1. Make sure paths for auxiliary files and results are specified correctly in SIFT_classes.py
2. Use "Main.ipynb" for example