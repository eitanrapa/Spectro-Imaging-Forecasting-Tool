import matplotlib.pyplot as plt
import corner
import pygtc
import numpy as np
from scipy.stats import median_abs_deviation
import h5py


class Projection:
    """
    A projection class that manages outputs and plots
    """

    def __init__(self, file_path):
        self.file_path = file_path

    def contour_plot_projection(self, file_name):
        """
        Plot in corner the contour plot of a specific run.
        :param file_name: name of file containing run
        :return: figure and data
        """

        # Read simulation output
        f = h5py.File(name=self.file_path + file_name, mode='r')

        data = f["chains"][:]

        # Create labels for contour plot
        labels = ('y', 'temperature', 'peculiar_velocity', 'a_sides', 'b_sides', 'CMB')
        y_value = f.attrs["y"]
        electron_temperature = f.attrs["electron_temperature"]
        peculiar_velocity = f.attrs["peculiar_velocity"]
        a_sides = f.attrs["a_sides"]
        b_sides = f.attrs["b_sides"]
        cmb_anis = f.attrs["cmb_anis"]
        theta = (y_value, electron_temperature, peculiar_velocity, a_sides, b_sides, cmb_anis)

        # Plot contour plot
        fig = corner.corner(
            data=data, labels=labels, truths=theta, smooth=1
        )
        plt.show()

        return fig, data

    def contour_plot_double_projection(self, file_name1, file_name2):
        """
        Plot using pygtc a comparison between two runs.
        :param file_name1: name of file containing first run
        :param file_name2: name of file containing second run
        :return: figure, data 1 and data2
        """

        # Read simulation output
        f1 = h5py.File(name=self.file_path + file_name1, mode='r')

        f2 = h5py.File(name=self.file_path + file_name2, mode='r')

        data1 = f1["chains"][:]
        data2 = f2["chains"][:]

        y_value = f1.attrs["y"]
        electron_temperature = f1.attrs["electron_temperature"]
        peculiar_velocity = f1.attrs["peculiar_velocity"]
        a_sides = f1.attrs["a_sides"]
        b_sides = f1.attrs["b_sides"]
        cmb_anis = f1.attrs["cmb_anis"]

        chain_labels = ["Run {}".format(str(file_name1)), "Run {}".format(str(file_name2))]

        labels = ('y', 'temperature', 'peculiar_vel', 'a_sides', 'b_sides', 'CMB')
        theta = (y_value, electron_temperature, peculiar_velocity, a_sides, b_sides, cmb_anis)

        # Plot contour plot
        gtc = pygtc.plotGTC(chains=[data1, data2], chainLabels=chain_labels, truths=theta, paramNames=labels,
                            figureSize=10)
        plt.show()

        return gtc, data1, data2

    def chain_projection(self, file_name):
        """
        Plots the MCMC chains of a run.
        :param file_name: name of file containing run
        :return: figure and axes
        """

        # Read simulation output
        f = h5py.File(name=self.file_path + file_name, mode='r')

        data = f["chains"][:]

        # Create labels for contour plot    # Create labels for contour plot
        labels = ('y', 'temperature', 'peculiar_velocity', 'a_sides', 'b_sides', 'CMB')

        fig, axes = plt.subplots(6, figsize=(30, 40), sharex=True)
        ndim = 6
        for i in range(ndim):
            ax = axes[i]
            ax.plot(data[:, i], "k", alpha=0.3)
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        plt.show()

        return fig, axes

    def statistics(self, file_name):
        """
        Get some statistics on the run
        :param file_name: name of file to access
        :return: mean and standard deviation of y, pec vel
        """

        # Read simulation output
        f = h5py.File(name=self.file_path + file_name, mode='r')

        data = f["chains"][:]

        y_mean = np.mean(data[:, 0])
        y_std = median_abs_deviation(data[:, 0])
        print("Mean of y_value = " + str(y_mean))
        print("Median of y_value = " + str(y_std))

        pec_vel_mean = median_abs_deviation(data[:, 2])
        pec_vel_std = median_abs_deviation(data[:, 2])
        print("Median of peculiar velocity = " + str(pec_vel_mean))
        print("Median of peculiar velocity = " + str(pec_vel_std))

        return y_mean, y_std, pec_vel_mean, pec_vel_std
