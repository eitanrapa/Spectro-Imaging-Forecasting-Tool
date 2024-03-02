import SIFT_classes as SIFT
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import corner
import pygtc
import numpy as np
import pickle

c = 299792458.0  # Speed of light - [c] = m/s
h_p = 6.626068e-34  # Planck's constant in SI units
k_b = 1.38065e-23  # Boltzmann constant in SI units
MJyperSrtoSI = 1e-20  # MegaJansky/Sr to SI units
GHztoHz = 1e9  # Gigahertz to hertz
HztoGHz = 1e-9  # Hertz to Gigahertz
TCMB = 2.725  # Canonical CMB in Kelvin
m = 9.109 * 10 ** (-31)  # Electron mass in kgs


def differential_intensity_projection(y_value, electron_temperature, peculiar_velocity, sides_long, sides_lat, bands,
                                      time):
    """
    Plots the spectral distortions of the galaxy cluster along with the CIB background and the instrument bands
    :param y_value: Y-value of galaxy cluster.
    :param electron_temperature: Electron temperature of galaxy cluster
    :param peculiar_velocity: Peculiar velocity of galaxy cluster
    :param sides_long: Longitudinal coordinates of SIDES catalog
    :param sides_lat: Latitudinal coordinates of SIDES catalog
    :param bands: Instrument bands, spectrometric or photometric
    :param time: Time of integration
    :return: None
    """

    # Create an arbitrary frequency space
    freq = np.linspace(80e9, 1000e9, 2000)

    # Get the main SZ distortion
    sz_template = SIFT.szpack_signal(freq, SIFT.y_to_tau(y_value, electron_temperature), electron_temperature,
                                     peculiar_velocity)

    # Sample the CIB from SIDES
    sides_template = SIFT.sides_continuum(freq, sides_long, sides_lat)

    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)

    # Plot SZ components
    plt.plot(freq * HztoGHz, abs(sz_template), '--k', label='Total SZ', linewidth=2)
    plt.plot(freq * HztoGHz,
             abs(SIFT.szpack_signal(freq, SIFT.y_to_tau(y_value, electron_temperature), electron_temperature,
                                    1e-11) - SIFT.classical_tsz(y_value, freq)),
             label='rSZ ' + str(electron_temperature) + ' keV')
    plt.plot(freq * HztoGHz, abs(SIFT.classical_tsz(y_value, freq)), label='tSZ y=' + str(y_value))

    # Plot the CIV
    plt.plot(freq * HztoGHz, abs(sides_template), color='pink', label='SIDES continuum')

    # Plot the instrument bands
    nu_total_array = np.empty(0)
    sigma_b_array = np.empty(0)

    for band in bands:
        if band['type'] == 'OLIMPO':

            # 80 hour normalized
            rms_value = band['rms'] * (np.sqrt(80) / (np.sqrt(time / 3600)))
            nu_vec_b = band['nu_meanGHz'] * GHztoHz
            x = h_p * nu_vec_b / (k_b * TCMB)
            sigma_B_b = 2 * k_b * ((nu_vec_b / c) ** 2) * (x / (np.exp(x) - 1)) * (x * np.exp(x)) / (
                    np.exp(x) - 1) * rms_value * 1e-6
            nu_total_array = np.concatenate((nu_total_array, nu_vec_b), axis=None)
            sigma_b_array = np.concatenate((sigma_b_array, sigma_B_b), axis=None)

        if band['type'] == 'spectrometric':
            nu_vec_b, sigma_B_b = SIFT.sig_b(band, time)
            nu_total_array = np.concatenate((nu_total_array, nu_vec_b), axis=None)
            sigma_b_array = np.concatenate((sigma_b_array, sigma_B_b), axis=None)

    plt.plot(nu_total_array * HztoGHz, sigma_b_array, 'o', lw=7, alpha=1, color='maroon')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('GHz', fontsize=20)
    plt.ylabel('W/m^2/Hz/Sr', fontsize=20)

    # Make xticks to match as best as possible
    plt.xticks(np.rint(np.logspace(np.log10(80), np.log10(1e3), num=9)))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, prop={'size': 12}, ncol=1,
               title='{} hour obs.'.format(time / 3600))
    figure(figsize=(20, 20), dpi=80)
    plt.show()
    return None


def contour_plot_projection(file_name):
    """
    Plot in corner the contour plot of a specific run.
    :param file_name: name of file containing run
    :return: figure and data
    """

    # Read simulation output
    data = np.load('/bolocam/bolocam/erapaport/new_runs/' + file_name, allow_pickle=True)

    # Read object
    with open('/bolocam/bolocam/erapaport/new_runs/' + file_name + '_object', 'rb') as f:
        sift_object = pickle.load(f)

    # Create labels for contour plot
    labels = ('y', 'temperature', 'peculiar_velocity', 'a_sides', 'b_sides', 'CMB')
    y_value = sift_object.y_value
    electron_temperature = sift_object.electron_temperature
    peculiar_velocity = sift_object.peculiar_velocity
    a_sides = sift_object.a_sides
    b_sides = sift_object.b_sides
    cmb_anis = sift_object.cmb_anis
    theta = (y_value, electron_temperature, peculiar_velocity, a_sides, b_sides, cmb_anis)

    # Plot contour plot
    fig = corner.corner(
        data, labels=labels, truths=theta, smooth=1
    )

    return fig, data


def contour_plot_double_projection(file_name1, file_name2):
    """
    Plot using pygtc a comparison between two runs.
    :param file_name1: name of file containing first run
    :param file_name2: name of file containing second run
    :return: figure, data 1 and data2
    """

    # Read simulation outputs
    data1 = np.load('/bolocam/bolocam/erapaport/new_runs/' + file_name1, allow_pickle=True)
    data2 = np.load('/bolocam/bolocam/erapaport/new_runs/' + file_name2, allow_pickle=True)

    # Read first object
    with open('/bolocam/bolocam/erapaport/new_runs/' + file_name1 + '_object', 'rb') as f:
        sift_object = pickle.load(f)

    # Create labels for contour plot
    y_value = sift_object.y_value
    electron_temperature = sift_object.electron_temperature
    peculiar_velocity = sift_object.peculiar_velocity
    a_sides = sift_object.a_sides
    b_sides = sift_object.b_sides
    cmb_anis = sift_object.cmb_anis

    chainLabels = ["Run {}".format(str(file_name1)), "Run {}".format(str(file_name2))]

    labels = ('y', 'temperature', 'peculiar_vel', 'a_sides', 'b_sides', 'CMB')
    theta = (y_value, electron_temperature, peculiar_velocity, a_sides, b_sides, cmb_anis)

    # Plot contour plot
    GTC = pygtc.plotGTC(chains=[data1, data2], chainLabels=chainLabels, truths=theta, paramNames=labels, figureSize=10)

    return GTC, data1, data2


def chain_projection(file_name):
    """
    Plots the MCMC chains of a run.
    :param file_name: name of file containing run
    :return: figure and axes
    """

    # Read simulation output, change directory
    data = np.load('/bolocam/bolocam/erapaport/new_runs/' + file_name, allow_pickle=True)

    # Create labels for contour plot    # Create labels for contour plot
    labels = ('y', 'temperature', 'peculiar_velocity', 'a_sides', 'b_sides', 'CMB')

    fig, axes = plt.subplots(5, figsize=(30, 40), sharex=True)
    ndim = 5
    for i in range(ndim):
        ax = axes[i]
        ax.plot(data[:, i], "k", alpha=0.3)
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    return fig, axes
