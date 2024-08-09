import matplotlib.pyplot as plt
import corner
import pygtc
import numpy as np
import pickle


class Projection:
    """
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
        data = np.load(file=self.file_path + file_name, allow_pickle=True)

        # # Read object
        # with open(file=self.file_path + file_name[:-4] + '_object', mode='rb') as f:
        #     sift_object = pickle.load(file=f)

        # Create labels for contour plot
        labels = ('y', 'temperature', 'peculiar_velocity', 'a_sides', 'b_sides', 'CMB')
        # y_value = sift_object.y_value
        # electron_temperature = sift_object.electron_temperature
        # peculiar_velocity = sift_object.peculiar_velocity
        # a_sides = sift_object.a_sides
        # b_sides = sift_object.b_sides
        # cmb_anis = sift_object.cmb_anis
        y_value = 5.4e-5
        electron_temperature = 5
        peculiar_velocity = 0
        a_sides = 1
        b_sides = 1
        cmb_anis = 0
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

        # Read simulation outputs
        data1 = np.load(self.file_path + file_name1, allow_pickle=True)
        data2 = np.load(self.file_path + file_name2, allow_pickle=True)

        # Read first object
        with open(file=self.file_path + file_name1[:-4] + '_object', mode='rb') as f:
            sift_object = pickle.load(file=f)

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
        GTC = pygtc.plotGTC(chains=[data1, data2], chainLabels=chainLabels, truths=theta, paramNames=labels,
                            figureSize=10)
        plt.show()

        return GTC, data1, data2

    def chain_projection(self, file_name):
        """
        Plots the MCMC chains of a run.
        :param file_name: name of file containing run
        :return: figure and axes
        """

        # Read simulation output, change directory
        data = np.load(self.file_path + file_name, allow_pickle=True)

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
