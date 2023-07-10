import BIAS_classes as BIAS
import numpy as np

# Pixels per band
# 3 Channel case
# Detector numbers taken from OLIMPO
pix1 = 50
pix2 = 136
pix3 = 282
pix4 = 460

# Band resolutions for SOFTS
# 3 Channel case
res1 = 17
res2 = 39
res3 = 26
res4 = 27
 
# NEPs for each band for SOFTS
NEP1 = np.sqrt(8.87**2 + 3.1**2)
NEP2 = np.sqrt(10.4**2 + 2.53**2)
NEP3 = np.sqrt(6.9**2 + 1.37**2)
NEP4 = np.sqrt(7.12**2 + 1.00**2) # all in aW/rtHz units

# Band definitions
# Center frequency and BW taken from OLIMPO
Bands_list_three_chan = [{'name':'Band 1','nu_meanGHz':145,'FBW':0.3,'nu_resGHz':res1,'NEP_aWrtHz':NEP1,'N_pixels':pix1},\
      {'name':'Band 2','nu_meanGHz':250,'FBW':0.4,'nu_resGHz':res2,'NEP_aWrtHz':NEP2,'N_pixels':pix2},\
      {'name':'Band 3','nu_meanGHz':365,'FBW':0.18,'nu_resGHz':res3,'NEP_aWrtHz':NEP3,'N_pixels':pix3},\
      {'name':'Band 4','nu_meanGHz':460,'FBW':0.15,'nu_resGHz':res4,'NEP_aWrtHz':NEP4,'N_pixels':pix4}]

# Band resolutions
# 5 Channel case
res1 = 8
res2 = 20
res3 = 12
res4 = 14

# Band definitions
# Center frequency and BW taken from OLIMPO
Bands_list_five_chan = [{'name':'Band 1','nu_meanGHz':145,'FBW':0.3,'nu_resGHz':res1,'NEP_aWrtHz':NEP1,'N_pixels':pix1},\
      {'name':'Band 2','nu_meanGHz':250,'FBW':0.4,'nu_resGHz':res2,'NEP_aWrtHz':NEP2,'N_pixels':pix2},\
      {'name':'Band 3','nu_meanGHz':365,'FBW':0.18,'nu_resGHz':res3,'NEP_aWrtHz':NEP3,'N_pixels':pix3},\
      {'name':'Band 4','nu_meanGHz':460,'FBW':0.15,'nu_resGHz':res4,'NEP_aWrtHz':NEP4,'N_pixels':pix4}]

# Band resolutions
# Hybrid 3-5 channel
res3 = 26
res4 = 14

# Band definitions
# Center frequency and BW taken from OLIMPO
Bands_list_three_five_chan = [{'name':'Band 1','nu_meanGHz':145,'FBW':0.3,'nu_resGHz':res1,'NEP_aWrtHz':NEP1,'N_pixels':pix1},\
      {'name':'Band 2','nu_meanGHz':250,'FBW':0.4,'nu_resGHz':res2,'NEP_aWrtHz':NEP2,'N_pixels':pix2},\
      {'name':'Band 3','nu_meanGHz':365,'FBW':0.18,'nu_resGHz':res3,'NEP_aWrtHz':NEP3,'N_pixels':pix3},\
      {'name':'Band 4','nu_meanGHz':460,'FBW':0.15,'nu_resGHz':res4,'NEP_aWrtHz':NEP4,'N_pixels':pix4}]

# List of cluster y-values to explore
y = [5.40E-05,1.80E-05,5.40E-05,1.80E-05,6.00E-06,2.00E-06,6.00E-06,2.00E-06,5.40E-05,1.80E-05,6.00E-06,2.00E-06,6.00E-06,2.00E-06,5.40E-05,1.80E-05,6.00E-06,2.00E-06,6.00E-06,2.00E-06,5.40E-05,1.80E-05,6.00E-06,2.00E-06,6.00E-06,2.00E-06]
     
# List of cluster electron temperatures to explore
electron_temperature = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 0.5, 0.5, 5.0, 5.0, 5.0, 5.0, 0.5, 0.5, 5.0, 5.0, 5.0, 5.0, 0.5, 0.5, 5.0, 5.0, 5.0, 5.0, 0.5, 0.5] #KeV   
     
# Constant fiducial velocity of 0
peculiar_vel = 0.000001 #km/s
betac = [peculiar_vel/(3e5)]
     
# Run names to save to
run_names_hybrid = ["425","426","405","406","407","408","411","412","401","402","403","404","409","410","413","414","415","416","417","418","419","420","421","422","423","424"]

# Integration times for each run
times = [1.728e6,1.728e6,1.152e6,1.152e6,1.152e6,1.152e6,1.152e6,1.152e6,288000,288000,288000,288000,288000,288000,144000,144000,144000,144000,144000,144000,72000,72000,72000,72000,72000,72000] #Seconds
     
# Loop over runs, define number of processors to use
for i in range(26):
#     biasclass = BIAS.Spectral_Simulation_SOFTS(y[i], electron_temperature[i], peculiar_vel, Bands_list_three_chan, times[i])
#     biasclass.run_sim(run_names_phot[i], "OLIMPO", processors_pool = 20)
    
#     biasclass = BIAS.Spectral_Simulation_SOFTS(y[i], electron_temperature[i], peculiar_vel, Bands_list_three_chan, times[i])
#     biasclass.run_sim(run_names_three[i], "SOFTS", processors_pool = 20)
    
#     biasclass = BIAS.Spectral_Simulation_SOFTS(y[i], electron_temperature[i], peculiar_vel, Bands_list_five_chan, times[i])
#     biasclass.run_sim(run_names_five[i], "SOFTS", processors_pool = 20)
    
    biasclass = BIAS.Spectral_Simulation_SOFTS(y[i], electron_temperature[i], peculiar_vel, Bands_list_three_five_chan, times[i])
    biasclass.run_sim(run_names_hybrid[i], "HYBRID", processors_pool = 20)
    
    
    

