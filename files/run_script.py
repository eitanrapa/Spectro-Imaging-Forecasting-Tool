import sift

hybrid_band = sift.bands.hybrid()

file_path = "/home/user/Documents/projects/runs/"
parameter_path = "/home/user/Documents/projects/runs/parameter_file_100.npy"
y_values = [5.4e-5]
electron_temperatures = [5.0]
peculiar_velocities = [1e-11]
times = [288000]
file_names = ['run_1']
bands = [hybrid_band]
realizations = 100
processors = 1

for i in range(len(y_values)):

    siftsim = sift.simulation(y_value=y_values[i], electron_temperature=electron_temperatures[i],
                              peculiar_velocity=peculiar_velocities[i], bands=bands[i], time=times[i])

    siftsim.run_sim(parameter_file=parameter_path, processors_pool=processors)

    siftsim.save(file_path=file_path, file_name=file_names[i])
