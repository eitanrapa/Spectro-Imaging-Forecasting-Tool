import sift

save_path = ""
parameter_file_path = ""
parameter_file = 'parameter_file_100.npy'

hybrid_band = sift.bands.hybrid()

y_values = [5.4e-5]
electron_temperatures = [5.0]
peculiar_velocities = [1e-11]
times = [288000]
file_names = ['run_2']
bands = [hybrid_band]
realizations = 100
processors = 1

for i in range(len(y_values)):
    
        sift_sim = sift.Simulation(y_value=y_values[i], electron_temperature=electron_temperatures[i], peculiar_velocity=peculiar_velocities[i], bands=bands[i], time=times[i])
        
        sift_sim.run_sim(realizations=realizations, processors=processors, file_path=parameter_file_path, parameter_file=parameter_file)
        
        sift_sim.save(file_path=save_path, file_name=file_names[i])
