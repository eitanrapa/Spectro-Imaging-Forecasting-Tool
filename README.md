# Bayesian Inference of Astrophysical imaging Spectrometers

Tool for forecasting Spectrometric and Photometric devices in the study of Sunyaev-Zeldovich distortions.
Includes CIB contamination with SIDES catalog, and CMB anisotropies as contaminants.
Resolves for y-value, electron temperature, peculiar velocity, and profile of CIB. 

Requires SZpack for use.
Also requires Pandas, AstroPy, NumPy, Matplotlib, Corner, Emcee, SciPy, multiprocessing.

Requires particular files:

sides.csv:

A file with the averaged distribution of all spectra from continuum.fits, from SIDES

parameter_file:

A file with some specific parameters for 40 randomized variations of the CMB, tSZ, kSZ anisotropies and CIB. 
tSZ, kSZ anisotropies are taken from calculations of the power spectra at 3 arcmin.
CMB values are taken from CMB.fits, taken from NASA website with real CMB values at an angular distance of 3 arcmin resolution.

continuum.fits:

Spectra of CIB from SIDES catalog