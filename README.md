# Code developed and maintained exclusively by Eitan Rapaport, Ritoban Basu Thakur
# Spectro-Imaging-Forecasting-Tool (SIFT)
Tool for forecasting Spectrometric and Photometric devices in the study of Sunyaev-Zeldovich distortions.
Includes CIB contamination with SIDES catalog, and CMB anisotropies as contaminants.
Resolves for y-value, electron temperature, peculiar velocity, profile of CIB, CMB anisotropy in an MCMC.

# Installation Instructions

1. Have/install GCC and GNU make
2. Have/install at least Python 3.7.2
3. Clone https://github.com/aivazis/mm.git
4. Clone https://github.com/pyre/pyre
5. Create mm config file:
    a. Go to home directory
    b. Go to or create a .config directory
    c. Create a directory called "mm"
    d. Copy and paste the following into a file named "config.mm" in the "mm" directory:
    
        # -*- Makefile -*-
        #

        # external dependencies
        # system tools
        sys.prefix := ${CONDA_PREFIX}

        # gsl
        gsl.version := 2.7
        gsl.dir := $(sys.prefix)

        # hdf5
        hdf5.version := 1.15.0
        hdf5.dir := ${sys.prefix}
        hdf5.parallel := off

        # libpq
        libpq.version := 16.1
        libpq.dir := ${sys.prefix}

        # python
        python.version := 3.12
        python.dir := $(sys.prefix)

        # pybind11
        pybind11.version := 2.11.1
        pybind11.dir = $(sys.prefix)

        # numpy
        numpy.version := 2.1
        numpy.dir := $(sys.prefix)/lib/python$(python.version)/site-packages/numpy/_core

        # pyre
        pyre.version := 1.12.4
        pyre.dir := $(sys.prefix)

        # install the python packages straight where they need to go
        builder.dest.pyc := $(sys.prefix)/lib/python$(python.version)/site-packages/

        # control over the build process
        # set the python compiler so we don't depend on the symbolic link, which may not even be there
        python3.driver := python$(python.version)
        
        # end of file

6. Create the mm yaml file:
    a. Go to home directory
    b. Go to .config directory
    c. Create a directory called "pyre"
    d. Copy and paste the following into a file named "mm.yaml" in the "pyre" directory:
    
        # -*- yaml -*-
        #

        # mm configuration
        mm:

          # targets
          target: opt, shared

          # compilers
          compilers: gcc, nvcc, python/python3

          # the following two settings get replaced with actual values by the notebook
          # the location of final products
          prefix: "{pyre.environ.CONDA_PREFIX}"
          # the location of the temporary intermediate build products
          bldroot: "{pyre.environ.HOME}/tmp/builds/mm/{pyre.environ.CONDA_DEFAULT_ENV}"
          # the installation location of the python packages
          pycPrefix: "lib/python3.12/site-packages"

          # misc
          # the name of GNU make
          make: make
          # local makefiles
          local: Make.mmm

        # end of file
        
7. Create a conda/mamba environment for the package
    a. Install conda/mamba if necessary
    b. Make a file in any directory and call it "sift.yaml"
    c. Copy and paste the following into the file:
    
        # -*- yaml -*-
        #


        name: SIFTenv

        channels:
          - conda-forge

            
            dependencies:
          - python
          - git
          - gcc
          - gxx
          - gfortran
          - make
          - automake
          - nodejs
          - libtool
          - curl
          - fftw
          - gsl
          - hdf5
          - libpq
          - openssl
          - pip
          - setuptools
          - graphene
          - matplotlib
          - numpy
          - pybind11
          - pytest
          - ruamel.yaml
          - scipy
          - yaml
          - pyyaml
          - ipython
          - pandas
          - astropy
          - camb=1.5.8
          - matplotlib
          - corner
          - emcee
          - tqdm
          - scipy
          - pygtc
          - h5py
          - gitpython

        # end of file

    d. Run the command "conda create -f sift.yaml"
    e. Activate the environment
    
8. Go to the pyre directory and run the command "python3 [PATH]/mm/mm.py" replacing [PATH] with the path to the mm directory
9. Go to the SIFT directory and run the command "python3 [PATH]/mm/mm.py" replacing [PATH] with the path to the mm directory
10. Ready to go!

# Usage instructions

1. Create a band input file and a simulation input file in the files directory, maintaining the structure like the examples
2. Create a differential intensity projection by typing "dip -run_file=[RUN_FILE].json -band_file=[BANDS_FILE].json", replacing [RUN_FILE] and [BANDS_FILE]
3. Run the simulation by typing "run -run_file=[RUN_FILE].json -band_file=[BANDS_FILE].json", replacing [RUN_FILE] and [BANDS_FILE]
4. Analyze the output by initializing a sift projection object:

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
