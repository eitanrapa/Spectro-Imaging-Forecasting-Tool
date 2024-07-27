import numpy as np
from astropy.io import fits
import pandas as pd
from scipy.interpolate import interp1d

# Make sure to append SZpack path
import sys
sys.path.append("/home/bolocam/erapaport/SIFT_old/codes/SZpack.v1.1.1/python")
import SZpack as SZ

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
    sides_template = interpolate(freq=freq, datax=np.linspace(0, 1500e9, 751), datay=total_SIDES)
    return sides_template


def sides_average(freq, a_sides, b_sides):
    """
    Get the average spectrum of the CIB across SIDES catalog modified by some shape parameters
    :param freq: Frequency space
    :param a_sides: Amplitude modulation of SIDES
    :param b_sides: Frequency modulation of SIDES
    """

    # Read SIDES average model
    df = pd.read_csv(filepath_or_buffer='/bolocam/bolocam/erapaport/Auxiliary/sides.csv', header=None)
    data = df.to_numpy()
    data = data.squeeze()
    SIDES = data * MJyperSrtoSI

    # Modify template by a_sides, b_sides
    sides_template = a_sides * interpolate(freq=freq, datax=np.linspace(0, 1500e9, 751) * b_sides, datay=SIDES)
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