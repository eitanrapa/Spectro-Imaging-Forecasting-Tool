import numpy as np
import pickle
import emcee
import matplotlib.pyplot as plt
from multiprocessing import Pool
import sift
from astropy.io import fits
import pandas as pd
from scipy.interpolate import interp1d
import git

# Constants
c = 299792458.0  # Speed of light - [c] = m/s
h_p = 6.626068e-34  # Planck's constant in SI units
k_b = 1.38065e-23  # Boltzmann constant in SI units
MJyperSrtoSI = 1e-20  # MegaJansky/Sr to SI units
GHztoHz = 1e9  # Gigahertz to hertz
HztoGHz = 1e-9  # Hertz to Gigahertz
TCMB = 2.725  # Canonical CMB in Kelvin
m = 9.109 * 10 ** (-31)  # Electron mass in kgs


def d_b(dt, frequency):
    """
    Blackbody function.
    :param dt: Differential temperature
    :param frequency: Frequency space
    :return: Spectral brightness distribution
    """

    temp = TCMB / (1 + dt)
    x = (h_p / (k_b * temp)) * frequency
    I = ((2 * h_p) / (c ** 2)) * (k_b * temp / h_p) ** 3
    return I * (x ** 4 * np.exp(x) / ((np.exp(x) - 1) ** 2)) * dt


def szpack_signal(frequency, tau, temperature, peculiar_velocity):
    """
    Use SZpack to create an SZ distortion distribution.
    :param frequency: Frequence space
    :param tau: Galaxy cluster optical depth
    :param temperature: Galaxy cluster electron temperature
    :param peculiar_velocity: Galaxy cluster peculiar velocity
    :return: Spectral brightness distribution
    """

    original_x_b = (h_p / (k_b * TCMB)) * frequency
    sz = sift.ext.szPack(tau=tau, temperature=temperature, peculiar_velocity=peculiar_velocity / 3e5)
    x_b = sz.sz_combo_means(original_x_b)
    return x_b * 13.33914078 * (TCMB ** 3) * (original_x_b ** 3) * MJyperSrtoSI


def classical_tsz(y, frequency):
    """
    Calculate the classical tSZ function.
    :param y: Galaxy cluster y-value
    :param frequency: Frequency space
    :return: Spectral brightness distribution
    """

    x_b = (h_p / (k_b * TCMB)) * frequency
    bv = 2 * k_b * TCMB * ((frequency ** 2) / (c ** 2)) * (x_b / (np.exp(x_b) - 1))
    return y * ((x_b * np.exp(x_b)) / (np.exp(x_b) - 1)) * (x_b * ((np.exp(x_b) + 1) / (np.exp(x_b) - 1)) - 4) * bv


def sides_continuum(freq, long, lat):
    """
    Get the spectrum of the CIB using the SIDES catalog.
    :param freq: Frequency space
    :param long: Longitudinal coordinates of SIDES
    :param lat: Longitudes coordinates of SIDES
    :return: Spectral brightness distortion
    """

    # Read FITS file
    repo = git.Repo('.', search_parent_directories=True)
    fname = repo.working_tree_dir + '/files/continuum.fits'
    hdu = fits.open(name=fname)
    image_data = hdu[0].data

    # SIDES spans 0 to 1500 GHz with 2 GHz intervals, with 0.5 arcmin resolution
    total_SIDES = np.zeros(shape=751)

    # Rebinning for 3 arcmin
    for col in range(6):
        for row in range(6):
            total_SIDES += image_data[:, long + row, lat + col] * MJyperSrtoSI
    total_SIDES = total_SIDES / 36

    # Interpolate for specific frequencies
    sides_template = interpolate(freq=freq, datax=np.linspace(2e9, 1500e9, 750), datay=total_SIDES[1:])
    return sides_template


def sides_average(freq, a_sides, b_sides):
    """
    Get the average spectrum of the CIB across SIDES catalog modified by some shape parameters
    :param freq: Frequency space
    :param a_sides: Amplitude modulation of SIDES
    :param b_sides: Frequency modulation of SIDES
    """

    # Read SIDES average model
    repo = git.Repo('.', search_parent_directories=True)
    df = pd.read_csv(filepath_or_buffer=repo.working_tree_dir + '/files/sides.csv', header=None)
    data = df.to_numpy()
    data = data.squeeze()
    SIDES = data * MJyperSrtoSI

    # Modify template by a_sides, b_sides
    sides_template = a_sides * interpolate(freq=freq, datax=np.linspace(2e9, 1500e9, 750) * b_sides,
                                           datay=SIDES[1:])
    return sides_template


def tau_to_y(tau, temperature):
    """
    Convert galaxy cluster optical depth to y-value
    :param tau: Galaxy cluster optical depth
    :param temperature: Galaxy cluster temperature
    :return: Galaxy Cluster y-value
    """

    if tau == 0:
        return 0
    return (tau * (temperature * 11604525.0061598) * k_b) / (m * (c ** 2))


def y_to_tau(y, temperature):
    """
    Convert galaxy cluster y-value to optical depth
    :param y: Galaxy cluster y-value
    :param temperature: Galaxy cluster temperature
    :return: Galaxy Cluster optical depth
    """

    if y == 0:
        return 0
    return (m * (c ** 2) * y) / (k_b * (temperature * 11604525.0061598))


def interpolate(freq, datax, datay):
    """
    Interpolate between data using a cubic spline
    :param freq: Frequency space
    :param datay: Y axis of data
    :param datax: X axis of data
    :return: Interpolated values at frequency space
    """

    f = interp1d(x=np.log(datax), y=np.log(datay), kind='slinear', bounds_error=False, fill_value=0)
    new_data = f(np.log(freq))
    return np.exp(new_data)


class Simulation:
    """
    This class defines the simulation suite. Initializing requires galaxy cluster parameters, instrument band
    definitions, and integration time for an observation.
    """

    def __init__(self, y_value, electron_temperature, peculiar_velocity, bands, time, a_sides=1, b_sides=1):
        self.y_value = y_value
        self.electron_temperature = electron_temperature
        self.peculiar_velocity = peculiar_velocity
        self.bands = bands
        self.time = time
        self.a_sides = a_sides
        self.b_sides = b_sides
        self.cmb_anis = 0
        self.data = None

    def differential_intensity_projection(self):
        """
        Plots the spectral distortions of the galaxy cluster along with the CIB background and the instrument bands
        :return: None
        """

        # Create an arbitrary frequency space
        freq = np.linspace(start=80e9, stop=1000e9, num=2000)

        # Get the main SZ distortion
        sz_template = szpack_signal(frequency=freq,
                                    tau=y_to_tau(y=self.y_value,
                                                 temperature=self.electron_temperature),
                                    temperature=self.electron_temperature,
                                    peculiar_velocity=self.peculiar_velocity)

        # Sample the CIB from SIDES
        sides_template = sides_average(freq=freq, a_sides=self.a_sides, b_sides=self.b_sides)

        plt.rc('xtick', labelsize=18)
        plt.rc('ytick', labelsize=18)

        # Plot SZ components
        plt.plot(freq * HztoGHz, abs(sz_template), '--k', label='Total SZ', linewidth=2)
        plt.plot(freq * HztoGHz,
                 abs(szpack_signal(frequency=freq,
                                   tau=y_to_tau(y=self.y_value,
                                                temperature=self.electron_temperature),
                                   temperature=self.electron_temperature,
                                   peculiar_velocity=1e-11) - classical_tsz(y=self.y_value,
                                                                            frequency=freq)),
                 label='rSZ ' + str(self.electron_temperature) + ' keV')
        plt.plot(freq * HztoGHz, abs(classical_tsz(y=self.y_value, frequency=freq)),
                 label='tSZ y=' + str(self.y_value))

        # Plot the CIB
        plt.plot(freq * HztoGHz, abs(sides_template), color='pink', label='CIB')

        # Plot the instrument bands
        nu_total_array, sigma_b_array = self.bands.get_sig_b(time=self.time)

        plt.plot(nu_total_array * HztoGHz, sigma_b_array, 'o', lw=7, alpha=1, color='maroon')

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('GHz', fontsize=20)
        plt.ylabel('W/m^2/Hz/Sr', fontsize=20)

        # Make xticks to match as best as possible
        plt.xticks(np.rint(np.logspace(np.log10(80), np.log10(1e3), num=9)))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, prop={'size': 12}, ncol=1,
                   title='{} hour obs.'.format(self.time / 3600))
        plt.show()

    def model(self, theta, freq):
        """
        Model of the MCMC used for the fit.
        :param theta: Values to constrain in the MCMC
        :param freq: Frequencies samples
        :return: Total spectral brightness distribution
        """

        y, temperature, peculiar_velocity, a_sides, b_sides, cmb_anis = theta

        # SIDES
        sides_template = sides_average(freq=freq, a_sides=a_sides, b_sides=b_sides)

        # SZ
        sz_template = szpack_signal(frequency=freq, tau=y_to_tau(y=y, temperature=temperature),
                                    temperature=temperature, peculiar_velocity=peculiar_velocity)

        # CMB
        cmb_template = d_b(dt=cmb_anis, frequency=freq)

        template_total = sz_template + sides_template + cmb_template

        return template_total

    def log_likelihood(self, theta, freq, data, noise):
        """
        Log likelihood function of MCMC.
        :param theta: Values to constrain in the MCMC
        :param freq: Frequencies samples
        :param data: Input fiducial observed data
        :param noise: Noise given by instrument
        :return: Log likelihood
        """

        model_data = self.model(theta=theta, freq=freq)
        return -0.5 * np.sum(((data - model_data) / noise) ** 2)

    def log_prior(self, theta):
        """
        Log prior function of MCMC
        :param theta: Values to constrain in the MCMC
        :return: Infinity if outside prior bounds, or a prior probability otherwise
        """

        y, temperature, peculiar_velocity, a_sides, b_sides, cmb_anis = theta
        betac = peculiar_velocity / 3e5

        # "Top-hat" prior for parameters
        if y < 0 or y > 1e-3:
            return -np.inf
        if betac < -0.03 or betac > 0.03:
            return -np.inf
        if temperature < 2.0 or temperature > 10.0:
            return -np.inf
        if a_sides < 0 or a_sides > 5.0:
            return -np.inf
        if b_sides < 0 or b_sides > 5.0:
            return -np.inf
        if cmb_anis < -1e-3 or cmb_anis > 1e-3:
            return -np.inf

        # Gaussian priors for CMB and galaxy cluster temp
        mu_temp = self.electron_temperature
        sigma_temp = self.electron_temperature / 20  # 5% precision

        mu_cmb = 0
        sigma_cmb = 110e-6  # 110 microKelvin

        # Convolved Gaussians in log space
        temp_gaussian = (1.0 / (sigma_temp * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((temperature - mu_temp) / sigma_temp) ** 2)
        cmb_gaussian = (1.0 / (sigma_cmb * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((cmb_anis - mu_cmb) / sigma_cmb) ** 2)

        return np.log(temp_gaussian) + np.log(cmb_gaussian)

    def log_probability(self, theta, freq, data, noise):
        """
        Log probability function of the MCMC.
        :param theta: Values to constrain in the MCMC
        :param freq: Frequencies samples
        :param data: Input fiducial observed data
        :param noise: Noise given by instrument
        :return: Log probability, cutting off infinite probabilities
        """

        lp = self.log_prior(theta=theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta=theta, freq=freq, data=data, noise=noise)

    def mcmc(self, anisotropies, long, lat, walkers, processors, chain_length):
        """
        Main MCMC calling function.
        :param anisotropies: Secondary CMB anisotropies from parameters
        :param long: Longitudinal coordinates of SIDES
        :param lat: Latitudinal coordinates of SIDES
        :param walkers: Number of walkers used for MCMC
        :param processors: Number of processors to use for MCMC, multiprocessing
        :param chain_length: Amount of samples to take for each chain
        :return: The sampler object from emcee with chains
        """

        ksz_anis, tsz_anis = anisotropies

        nu_total_array, sigma_b_array = self.bands.get_sig_b(time=self.time)

        # SIDES
        sides_template = sides_continuum(freq=nu_total_array, long=long, lat=lat)

        # SZ
        sz_template = szpack_signal(frequency=nu_total_array,
                                    tau=y_to_tau(y=self.y_value,
                                                 temperature=self.electron_temperature),
                                    temperature=self.electron_temperature,
                                    peculiar_velocity=self.peculiar_velocity)

        # CMB
        cmb_template = d_b(dt=self.cmb_anis, frequency=nu_total_array)

        # Secondary anisotropies
        tsz_template = classical_tsz(y=tsz_anis, frequency=nu_total_array)
        ksz_template = d_b(dt=ksz_anis, frequency=nu_total_array)

        total_sz_array = sz_template + sides_template + tsz_template + ksz_template + cmb_template

        # Define and run MCMC
        theta = self.y_value, self.electron_temperature, self.peculiar_velocity, self.a_sides, self.b_sides, \
            self.cmb_anis
        pos = []
        for item in theta:
            pos.append(item * (1 + 0.01 * np.random.randn(walkers)))
        pos_array = np.asarray(pos)
        pos_array = pos_array.transpose()
        nwalkers, ndim = pos_array.shape

        with Pool(processors) as pool:
            sampler = emcee.EnsembleSampler(nwalkers=nwalkers, ndim=ndim, log_prob_fn=self.log_probability,
                                            args=(nu_total_array, total_sz_array, sigma_b_array), pool=pool)
            for sample in sampler.sample(pos_array, iterations=chain_length, progress=True):
                continue

        return sampler

    def run_sim(self, parameter_file, chain_length=12000, walkers=100, realizations=100, discard_n=2000,
                thin_n=200, processors_pool=30):
        """
        Function used to analyze parameters and chains after calling MCMC.
        :param file_path: Where to save run
        :param parameter_file: File containing parameters for run
        :param walkers: Number of walkers used for MCMC
        :param processors_pool: Number of processors to use for MCMC, multiprocessing
        :param chain_length: Amount of samples to take for each chain
        :param realizations: Realizations which to concatenate
        :param discard_n: Discard first n from chain of MCMC
        :param thin_n: Discard every n from chain of MCMC
        :return None, saves run.
        """

        # Read saved parameters file
        params = np.load(file=parameter_file, allow_pickle=True)

        samples = [[0, 0, 0, 0, 0, 0]]
        samples = np.asarray(samples)

        for i in range(realizations):
            sides_long = int(params[i, 0])
            sides_lat = int(params[i, 1])
            self.cmb_anis = params[i, 2]
            amp_ksz = params[i, 3]
            amp_tsz = params[i, 4]
            anisotropies = (amp_ksz, amp_tsz)
            sampler = self.mcmc(anisotropies=anisotropies, long=sides_long, lat=sides_lat, walkers=walkers,
                                processors=processors_pool, chain_length=chain_length)
            samples = np.append(arr=samples, values=sampler.get_chain(discard=discard_n, flat=True, thin=thin_n),
                                axis=0)

        samples = samples[1:, :]

        self.data = samples

    def save(self, file_path, file_name):

        # Write simulation output
        np.save(file=file_path + file_name, arr=self.data, allow_pickle=True)

        # Write simulation data
        with open(file=file_path + file_name + '_object', mode='wb') as f:
            pickle.dump(obj=self, file=f)
