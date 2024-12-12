import matplotlib.pyplot as plt
import corner
import pygtc
import numpy as np
from scipy.stats import median_abs_deviation
import h5py
import warnings


warnings.filterwarnings(action="ignore", module="matplotlib")


class Projection:
    """
    A projection class that manages outputs and plots
    """

    def __init__(self, file_path):
        self.file_path = file_path

    def remove_outlier_simulations(self, data, target_num, n_sublists, n_remove):

        def remove_furthest_sublists(sublists, target_num, n_remove):

            # Compute the average of each sublist
            averages = [np.mean(sublist) for sublist in sublists]

            # Calculate the absolute difference between each average and the target number
            differences = [abs(avg - target_num) for avg in averages]

            # Sort the indices of the sublists based on differences, in descending order
            sorted_indices = np.argsort(differences)[::-1]

            # Indices of the sublists to keep (remove the first n_remove elements)
            remaining_indices = sorted_indices[n_remove:]

            # Return the remaining indices and the remaining sublists
            remaining_sublists = [sublists[i] for i in remaining_indices]

            return remaining_sublists, remaining_indices

        # Step 3: Define the function to remove corresponding sublists from all rows
        def remove_sublists_from_all_rows(data, target_num, n_sublists, n_remove):

            # Split all rows of the array into sublists
            split_data = [np.array_split(row, n_sublists) for row in data]

            # Remove sublists from the first row
            remaining_sublists_first_row, remaining_indices = remove_furthest_sublists(split_data[0], target_num,
                                                                                       n_remove)

            # Now remove corresponding sublists from all other rows based on remaining_indices
            remaining_sublists_all_rows = []
            for row_sublists in split_data:
                remaining_sublists = [row_sublists[i] for i in remaining_indices]
                remaining_sublists_all_rows.append(remaining_sublists)

            return remaining_sublists_all_rows

        # Step 4: Define the function to recombine sublists into a full 2D array
        def recombine_sublists(remaining_sublists_all_rows):

            # Flatten and concatenate the sublists in each row
            recombined_rows = [np.concatenate(row_sublists) for row_sublists in remaining_sublists_all_rows]

            # Combine all rows to form a 2D array
            recombined_array = np.vstack(recombined_rows)

            return recombined_array

        remaining_sublists_all_rows = remove_sublists_from_all_rows(data.T, target_num=target_num,
                                                                    n_sublists=n_sublists, n_remove=n_remove)

        # Recombine the remaining sublists into a 2D array
        recombined_data = recombine_sublists(remaining_sublists_all_rows)

        return recombined_data.T

    def contour_plot_projection(self, file_name, remove_outlier_simulations=0):
        """
        Plot in corner the contour plot of a specific run.
        :param file_name: name of file containing run
        :param remove_outlier_simulations: how many outlier sims to remove
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
        realizations = f.attrs["realizations"]

        theta = (y_value, electron_temperature, peculiar_velocity, a_sides, b_sides)

        data = self.remove_outlier_simulations(data, n_sublists=realizations, target_num=y_value,
                                               n_remove=remove_outlier_simulations)

        # Plot contour plot
        fig = corner.corner(
            data=data, labels=labels, truths=theta, smooth=1
        )
        plt.show()

        return fig, data

    def contour_plot_double_projection(self, file_name1, file_name2, remove_outlier_simulations=0):
        """
        Plot using pygtc a comparison between two runs.
        :param file_name1: name of file containing first run
        :param file_name2: name of file containing second run
        :param remove_outlier_simulations: how many outlier sims to remove
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
        realizations = f1.attrs["realizations"]

        data1 = self.remove_outlier_simulations(data1, n_sublists=realizations, target_num=y_value,
                                                n_remove=remove_outlier_simulations)

        data2 = self.remove_outlier_simulations(data2, n_sublists=realizations, target_num=y_value,
                                                n_remove=remove_outlier_simulations)

        chain_labels = ["Run {}".format(str(file_name1)), "Run {}".format(str(file_name2))]

        labels = ('y', 'temperature', 'peculiar_vel', 'a_sides', 'b_sides')
        theta = (y_value, electron_temperature, peculiar_velocity, a_sides, b_sides)

        # Plot contour plot
        gtc = pygtc.plotGTC(chains=[data1, data2], chainLabels=chain_labels, truths=theta, paramNames=labels,
                            figureSize=10)
        plt.show()

        return gtc, data1, data2

    def chain_projection(self, file_name, remove_outlier_simulations=0):
        """
        Plots the MCMC chains of a run.
        :param file_name: name of file containing run
        :param remove_outlier_simulations: how many outlier sims to remove
        :return: figure and axes
        """

        # Read simulation output
        f = h5py.File(name=self.file_path + file_name, mode='r')

        data = f["chains"][:]

        # Create labels for contour plot    # Create labels for contour plot
        labels = ('y', 'temperature', 'peculiar_velocity', 'a_sides', 'b_sides')

        realizations = f.attrs["realizations"]
        y_value = f.attrs["y"]

        data = self.remove_outlier_simulations(data, n_sublists=realizations, target_num=y_value,
                                               n_remove=remove_outlier_simulations)

        fig, axes = plt.subplots(4, figsize=(30, 40), sharex=True)
        ndim = 5
        for i in range(ndim):
            ax = axes[i]
            ax.plot(data[:, i], "k", alpha=0.3)
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        plt.show()

        return fig, axes

    def statistics(self, file_name, remove_outlier_simulations=0):
        """
        Get some statistics on the run
        :param file_name: name of file to access
        :param remove_outlier_simulations: how many outlier sims to remove
        :return: mean and standard deviation of y, pec vel
        """

        # Read simulation output
        f = h5py.File(name=self.file_path + file_name, mode='r')

        data = f["chains"][:]

        realizations = f.attrs["realizations"]
        y_value = f.attrs["y"]

        data = self.remove_outlier_simulations(data, n_sublists=realizations, target_num=y_value,
                                               n_remove=remove_outlier_simulations)

        y_median = np.median(data[:, 0])
        y_std = median_abs_deviation(data[:, 0])
        print("Median of y_value = " + str(y_median))
        print("Standard deviation of y_value = " + str(y_std))

        pec_vel_median = np.median(data[:, 2])
        pec_vel_std = median_abs_deviation(data[:, 2])
        print("Median of peculiar velocity = " + str(pec_vel_median))
        print("Standard deviation of peculiar velocity = " + str(pec_vel_std))

        return y_median, y_std, pec_vel_median, pec_vel_std
