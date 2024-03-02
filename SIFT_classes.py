import numpy as np
import sys
import pandas as pd
import pickle
import emcee
from astropy.io import fits
from scipy.interpolate import interp1d
from multiprocessing import Pool

# Make sure to append SZpack path
sys.path.append("/home/bolocam/erapaport/SIFT_old/codes/SZpack.v1.1.1/python")
sys.path.append("/bolocam/bolocam/erapaport/Auxiliary/")
import Mather_photonNEP12a as NEP
import SZpack as SZ


c = 299792458.0  # Speed of light - [c] = m/s
h_p = 6.626068e-34  # Planck's constant in SI units
k_b = 1.38065e-23  # Boltzmann constant in SI units
MJyperSrtoSI = 1e-20  # MegaJansky/Sr to SI units
GHztoHz = 1e9  # Gigahertz to hertz
HztoGHz = 1e-9  # Hertz to Gigahertz
TCMB = 2.725  # Canonical CMB in Kelvin
m = 9.109 * 10 ** (-31)  # Electron mass in kgs


# def sigB(band_details, time):
#     """
#     Noise Equivalent Brightness function with known NEPs
#     """
#
#     BW_GHz = band_details['nu_meanGHz'] * band_details['FBW']
#
#     nu_min = (band_details['nu_meanGHz'] - 0.5 * BW_GHz) * GHztoHz
#     nu_max = (band_details['nu_meanGHz'] + 0.5 * BW_GHz) * GHztoHz
#     nu_res = band_details['nu_resGHz'] * GHztoHz
#     Npx = band_details['N_pixels']
#
#     NEP_tot = (band_details['NEP_aWrtHz']) * 1E-18
#     Nse = int(np.round(BW_GHz / band_details['nu_resGHz']))
#     nu_vec = np.linspace(nu_min, nu_max, Nse)
#     AOnu = (c / nu_vec) ** 2
#
#     # Defined empirically to match OLIMPO inefficiencies at single channel bands
#     inefficiency = 0.019
#     delP = 2.0 * NEP_tot / np.sqrt(time * Npx)
#     sigma_B = delP / AOnu / nu_res / inefficiency
#
#     return nu_vec, sigma_B

def sig_b(band_details, time, tnoise=3.0):
    """
    Noise Equivalent Brightness function with unknown NEPs.
    Use for apples to apples with OLIMPO photometric mode.
    :param band_details: Bands of instrument
    :param time: Integration time
    :param tnoise: Thermal noise of CMB
    """

    BW_GHz = band_details['nu_meanGHz'] * band_details['FBW']

    nu_min = (band_details['nu_meanGHz'] - 0.5 * BW_GHz) * GHztoHz
    nu_max = (band_details['nu_meanGHz'] + 0.5 * BW_GHz) * GHztoHz
    nu_res = band_details['nu_resGHz'] * GHztoHz
    Npx = band_details['N_pixels']

    NEP_phot1 = NEP.photonNEPdifflim(nu_min, nu_max, tnoise)  # This is CMB Tnoise
    NEP_phot2 = NEP.photonNEPdifflim(nu_min, nu_max, 10.0, aef=0.01)  # Use real South Pole data
    NEP_det = 10e-18  # ATTO WATTS per square-root(hz)
    NEP_tot = np.sqrt(NEP_phot1 ** 2 + NEP_phot2 ** 2 + NEP_det ** 2)  # Don't include atmosphere for now

    # in making nu_vec we must be aware of resolution
    Nse = int(np.round(BW_GHz / band_details['nu_resGHz']))
    nu_vec = np.linspace(nu_min, nu_max, Nse)
    AOnu = (c / nu_vec) ** 2

    # Defined empirically to match OLIMPO inefficiencies at single channel bands
    inefficiency = 0.019
    delP = 2.0 * NEP_tot / np.sqrt(time * Npx)
    sigma_B = delP / AOnu / nu_res / inefficiency

    return nu_vec, sigma_B


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

    x_b = (h_p / (k_b * TCMB)) * frequency
    original_x_b = (h_p / (k_b * TCMB)) * frequency
    SZ.compute_combo_means(x_b, tau, temperature, peculiar_velocity / 3e5, 0, 0, 0, 0)
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
    fname = '/bolocam/bolocam/erapaport/Auxiliary/continuum.fits'
    hdu = fits.open(fname)
    image_data = hdu[0].data

    # SIDES spans 0 to 1500 GHz with 2 GHz intervals, with 0.5 arcmin resolution
    total_SIDES = np.zeros(751)

    # Rebinning for 3 arcmin
    for col in range(6):
        for row in range(6):
            total_SIDES += image_data[:, long + row, lat + col] * MJyperSrtoSI
    total_SIDES = total_SIDES / 36

    # Interpolate for specific frequencies
    sides_template = interpolate(freq, total_SIDES, np.linspace(0, 1500e9, 751))
    return sides_template


def sides_average(freq, a_sides, b_sides):
    """
    Get the average spectrum of the CIB across SIDES catalog modified by some shape parameters
    :param freq: Frequency space
    :param a_sides: Amplitude modulation of SIDES
    :param b_sides: Frequency modulation of SIDES
    """

    # Read SIDES average model
    df = pd.read_csv('/bolocam/bolocam/erapaport/Auxiliary/sides.csv', header=None)
    data = df.to_numpy()
    data = data.squeeze()
    SIDES = data * MJyperSrtoSI

    # Modify template by a_sides, b_sides
    sides_template = a_sides * interpolate(freq, SIDES, np.linspace(0, 1500e9, 751) * b_sides)
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


def interpolate(freq, datay, datax):
    """
    Interpolate between data using a cubic spline
    :param freq: Frequency space
    :param datay: Y axis of data
    :param datax: X axis of data
    :return: Interpolated values at frequency space
    """

    f = interp1d(np.log(datax), np.log(datay), kind='cubic', bounds_error=False, fill_value=0)
    new_data = f(np.log(freq))
    return np.exp(new_data)


class SpectralSimulation:
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

    def model(self, theta, freq):
        """
        Model of the MCMC used for the fit.
        :param theta: Values to constrain in the MCMC
        :param freq: Frequencies samples
        :return: Total spectral brightness distribution
        """

        y, temperature, peculiar_velocity, a_sides, b_sides, cmb_anis = theta

        # SIDES
        sides_template = sides_average(freq, a_sides, b_sides)

        # SZ
        sz_template = szpack_signal(freq, y_to_tau(y, temperature), temperature, peculiar_velocity)

        # CMB
        cmb_template = d_b(cmb_anis, freq)

        template_total = sz_template + sides_template + cmb_template

        return template_total

    def templates(self, freq, long, lat):
        """
        Separated individual total templates of the MCMC.
        :param freq: Frequencies sampled
        :param long: Longitudinal coordinates of SIDES
        :param lat: Latitudinal coordinates of SIDES
        :return: Separated individual spectral brightness distributions
        """

        # SIDES
        sides_template = sides_continuum(freq, long, lat)

        # Galaxy cluster SZ template
        sz_template = szpack_signal(freq, y_to_tau(self.y_value, self.electron_temperature), self.electron_temperature,
                                    self.peculiar_velocity)

        # CMB
        cmb_template = d_b(self.cmb_anis, freq)

        template_total = [sz_template, sides_template, cmb_template]
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

        model_data = self.model(theta, freq)
        return -0.5 * np.sum(((data - model_data) / noise) ** 2)

    def log_prior(self, theta):
        """
        Log prior function of MCMC
        :param theta: Values to constrain in the MCMC
        :return: Infinity if outside prior bounds, or a prior probability otherwise
        """

        y, temperature, peculiar_velocity, a_sides, b_sides, cmb_anis = theta

        # "Top-hat" prior for parameters
        if y < 0 or y > 0.1:
            return -np.inf
        if (peculiar_velocity / 3e5) < -0.02 or (peculiar_velocity / 3e5) > 0.02:
            return -np.inf
        if temperature < 2.0 or temperature > 75.0:
            return -np.inf
        if a_sides < 0 or a_sides > 2.5:
            return -np.inf
        if b_sides < 0 or b_sides > 2.5:
            return -np.inf
        if cmb_anis < -1e-3 or cmb_anis > 1e-3:
            return -np.inf

        # Gaussian priors for CMB and galaxy cluster temp
        mu_temp = self.electron_temperature
        sigma_temp = self.electron_temperature / 10  # 10% precision

        mu_cmb = self.cmb_anis
        sigma_cmb = self.cmb_anis / 10  # 10% precision

        # Convolved Gaussians in log space
        return np.log(1.0 / (np.sqrt(2 * np.pi) * sigma_temp)) - 0.5 * (
                temperature - mu_temp) ** 2 / sigma_temp ** 2 + np.log(
            1.0 / (np.sqrt(2 * np.pi) * sigma_cmb)) - 0.5 * (cmb_anis - mu_cmb) ** 2 / sigma_cmb ** 2

    def log_probability(self, theta, freq, data, noise):
        """
        Log probability function of the MCMC.
        :param theta: Values to constrain in the MCMC
        :param freq: Frequencies samples
        :param data: Input fiducial observed data
        :param noise: Noise given by instrument
        :return: Log probability, cutting off infinite probabilities
        """

        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta, freq, data, noise)

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

        # Analyze instrument bands
        nu_total_array = np.empty(0)
        sigma_b_array = np.empty(0)

        for band in self.bands:
            if band['type'] == 'OLIMPO':

                # 80 hour normalized
                rms_value = band['rms'] * (np.sqrt(80) / (np.sqrt(self.time / 3600)))
                nu_vec_b = band['nu_meanGHz'] * GHztoHz
                x = h_p * nu_vec_b / (k_b * TCMB)
                sigma_B_b = 2 * k_b * ((nu_vec_b / c) ** 2) * (x / (np.exp(x) - 1)) * (x * np.exp(x)) / (
                        np.exp(x) - 1) * rms_value * 1e-6
                nu_total_array = np.concatenate((nu_total_array, nu_vec_b), axis=None)
                sigma_b_array = np.concatenate((sigma_b_array, sigma_B_b), axis=None)

            if band['type'] == 'spectrometric':

                nu_vec_b, sigma_B_b = sig_b(band, self.time)
                nu_total_array = np.concatenate((nu_total_array, nu_vec_b), axis=None)
                sigma_b_array = np.concatenate((sigma_b_array, sigma_B_b), axis=None)

        # SIDES
        sides_template = sides_continuum(nu_total_array, long, lat)

        # SZ
        sz_template = szpack_signal(nu_total_array, y_to_tau(self.y_value, self.electron_temperature),
                                    self.electron_temperature, self.peculiar_velocity)

        # CMB
        cmb_template = d_b(self.cmb_anis, nu_total_array)

        # Secondary anisotropies
        tsz_template = classical_tsz(tsz_anis, nu_total_array)
        ksz_template = d_b(ksz_anis, nu_total_array)

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
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability,
                                            args=(nu_total_array, total_sz_array, sigma_b_array), pool=pool)
            for sample in sampler.sample(pos_array, iterations=chain_length, progress=True):
                continue

        return sampler

    def run_sim(self, file_name, parameter_file, chain_length=12000, walkers=100, realizations=500, discard_n=2000,
                thin_n=200, processors_pool=30):
        """
        Function used to analyze parameters and chains after calling MCMC.
        :param file_name: Name of where to save run
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
        params = np.load('/bolocam/bolocam/erapaport/Auxiliary/' + parameter_file, allow_pickle=True)

        samples = [[0, 0, 0, 0, 0, 0]]
        samples = np.asarray(samples)

        for i in range(realizations):
            sides_long = int(params[i, 0])
            sides_lat = int(params[i, 1])
            self.cmb_anis = params[i, 2]
            amp_ksz = params[i, 3]
            amp_tsz = params[i, 4]
            anisotropies = (amp_ksz, amp_tsz)
            sampler = self.mcmc(anisotropies, sides_long, sides_lat, walkers, processors_pool, chain_length)
            samples = np.append(samples, sampler.get_chain(discard=discard_n, flat=True, thin=thin_n), axis=0)

        samples = samples[1:, :]

        # Write simulation output
        np.save('/bolocam/bolocam/erapaport/new_runs/' + file_name, samples, allow_pickle=True)

        # Write simulation data
        with open('/bolocam/bolocam/erapaport/new_runs/' + file_name + '_object', 'wb') as f:
            pickle.dump(self, f)
