# Code developed and maintained exclusively by Eitan Rapaport, Ritoban Basu Thakur
# Spectro-Imaging-Forecasting-Tool (SIFT)
Tool for forecasting Spectrometric and Photometric devices in the study of Sunyaev-Zeldovich distortions.
Includes CIB contamination with SIDES catalog, and CMB anisotropies as contaminants.
Resolves for y-value, electron temperature, peculiar velocity, profile of CIB, CMB anisotropy in an MCMC.

Requires python packages: Pandas, AstroPy, CAMB, NumPy, Matplotlib, Corner, Emcee, SciPy, multiprocessing, pygtc,
matplotlib, PyBind11, h5py

# Installation Instructions

1. Install GSL
2. Download https://github.com/aivazis/mm.git

# Usage instructions

1. Create a band input file and a simulation input file, maintaining the structure like the examples
2. Create a differential intensity projection by typing
2. Run the simulation by typing 
3. Analyze the output by initializing a sift projection object:

```
import sift

# Create a projection object with the path to the folder with runs
siftproj = sift.projection('PATH_TO_RUNS')

# Retrieve a contour plot
fig, data = siftproj.contour_plot_projection('run_1.hdf5')

# Look at the chains
fig, axes = siftproj.chain_projection('run_1.hdf5')

# Look at the double projection of two runs
gtc, data1, data2 = siftproj.contour_plot_double_projection('run_1.hdf5', 'run_2.hdf5')

# Get the statistics on a run
y_mean, y_std, pec_vel_mean, pec_vel_std = siftproj.statistics('run_1.hdf5')
```