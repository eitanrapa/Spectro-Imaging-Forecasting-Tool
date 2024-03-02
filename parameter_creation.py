import numpy as np
import camb

seed = 583
np.random.seed(seed)


def create_cmb_map(resolution=3.0):
    """
    Creates a map of the CMB anisotropies using CAMB.
    :param resolution: resolution of the map in arcmin
    :return: a cmb map in Kelvin
    """

    # Set up a new set of parameters for CAMB
    pars = camb.CAMBparams()

    # This function sets up with one massive neutrino and helium set using BBN consistency
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
    pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
    lmax = 5000
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)

    # calculate results for these parameters
    results = camb.get_results(pars)

    # get dictionary of CAMB power spectra
    powers = results.get_cmb_power_spectra(pars, CMB_unit='K')
    totCL = powers['total']
    DlTT = totCL[:, 0]
    ell = np.arange(totCL.shape[0])

    # variables to set up the size of the map
    N = 2 ** 10  # this is the number of pixels in a linear dimension

    # since we are using lots of FFTs this should be a factor of 2^N
    pix_size = resolution  # size of a pixel in arcminutes

    # convert Dl to Cl
    ClTT = DlTT * 2 * np.pi / (ell * (ell + 1.))
    ClTT[0] = 0.  # set the monopole and the dipole of the Cl spectrum to zero
    ClTT[1] = 0.

    # make a 2D real space coordinate system
    onesvec = np.ones(N)
    inds = (np.arange(N) + .5 - N / 2.) / (N - 1.)  # create an array of size N between -0.5 and +0.5

    # compute the outer product matrix: X[i, j] = onesvec[i] * inds[j] for i,j
    # in range(N), which is just N rows copies of inds - for the x dimension
    X = np.outer(onesvec, inds)

    # compute the transpose for the y dimension
    Y = np.transpose(X)

    # radial component R
    R = np.sqrt(X ** 2. + Y ** 2.)

    # now make a 2D CMB power spectrum
    pix_to_rad = (
            pix_size / 60. * np.pi / 180.)  # going from pix_size in arcmins to degrees and then degrees to radians
    ell_scale_factor = 2. * np.pi / pix_to_rad  # now relating the angular size in radians to multipoles
    ell2d = R * ell_scale_factor  # making a fourier space analogue to the real space R vector
    ClTT_expanded = np.zeros(int(ell2d.max()) + 1)

    # making an expanded Cl spectrum (of zeros) that goes all the way to the size of the 2D ell vector
    ClTT_expanded[0:ClTT.size] = ClTT  # fill in the Cls until the max of the ClTT vector

    # the 2D Cl spectrum is defined on the multiple vector set by the pixel scale
    CLTT2d = ClTT_expanded[ell2d.astype(int)]

    # now make a realization of the CMB with the given power spectrum in real space
    random_array_for_T = np.random.normal(0, 1, (N, N))
    FT_random_array_for_T = np.fft.fft2(random_array_for_T)  # take FFT since we are in Fourier space

    FT_2d = np.sqrt(CLTT2d) * FT_random_array_for_T  # we take the sqrt since the power spectrum is T^2

    # move back from ell space to real space
    CMB_T = np.fft.ifft2(np.fft.fftshift(FT_2d))
    # move back to pixel space for the map
    CMB_T = CMB_T / (pix_size / 60. * np.pi / 180.)
    # we only want to plot the real component
    CMB_T = np.real(CMB_T)

    return CMB_T


def create_parameter_file(cmb_resolution, realizations):
    """
    Creates a file of fiducial parameters for contaminants including CMB, CIB, and secondary tsz and ksz.
    :param cmb_resolution: resolution of the CMB anisotropy sampling in arcmin
    :param realizations: amount of realizations
    """

    params = [[0, 0, 0, 0, 0]]
    params = np.asarray(params)

    # Create CMB map
    CMB_T = create_cmb_map(cmb_resolution)

    # Make realizations
    for i in range(realizations):

        # Pick coordinates of SIDES continuum
        sides_long = np.random.randint(0, 160)
        sides_lat = np.random.randint(0, 160)

        # Pick coordinates of CMB map
        cmb_long = np.random.randint(2, 1000)
        cmb_lat = np.random.randint(2, 1000)

        # Inpainting of 5x5
        amp_cmb = CMB_T[cmb_long, cmb_lat] * 1e-6
        amp_cmb = amp_cmb - np.mean(CMB_T[cmb_long - 2:cmb_long + 3, cmb_lat - 2:cmb_lat + 3])

        # Make CMB secondary anisotropies
        # Numbers determined empirically
        amp_ksz = np.random.normal(0, 6e-7)
        amp_tsz = np.random.normal(0, 6e-7)

        params_realization = [[sides_long, sides_lat, amp_cmb, amp_ksz, amp_tsz]]
        params = np.append(params, params_realization, axis=0)

    params = params[1:, :]

    # Write simulation output, change directory/name
    np.save('/bolocam/bolocam/erapaport/Auxiliary/parameter_file_' + str(realizations), params, allow_pickle=True)
