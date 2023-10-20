import SIFT_classes as SIFT
import numpy as np

# # Pixels per band
# # 3 Channel case
# # Detector numbers taken from OLIMPO
# pix1 = 50
# pix2 = 136
# pix3 = 282
# pix4 = 460

# # Band resolutions for SOFTS
# # 3 Channel case
# res1 = 17
# res2 = 39
# res3 = 26
# res4 = 27
 
# # NEPs for each band for SOFTS
# NEP1 = np.sqrt(8.87**2 + 3.1**2)
# NEP2 = np.sqrt(10.4**2 + 2.53**2)
# NEP3 = np.sqrt(6.9**2 + 1.37**2)
# NEP4 = np.sqrt(7.12**2 + 1.00**2) # all in aW/rtHz units

# # Band definitions
# # Center frequency and BW taken from OLIMPO
# Bands_list_three_chan = [{'name':'Band 1','nu_meanGHz':145,'FBW':0.3,'nu_resGHz':res1,'NEP_aWrtHz':NEP1,'N_pixels':pix1},\
#       {'name':'Band 2','nu_meanGHz':250,'FBW':0.4,'nu_resGHz':res2,'NEP_aWrtHz':NEP2,'N_pixels':pix2},\
#       {'name':'Band 3','nu_meanGHz':365,'FBW':0.18,'nu_resGHz':res3,'NEP_aWrtHz':NEP3,'N_pixels':pix3},\
#       {'name':'Band 4','nu_meanGHz':460,'FBW':0.15,'nu_resGHz':res4,'NEP_aWrtHz':NEP4,'N_pixels':pix4}]

# # Band resolutions
# # 5 Channel case
# res1 = 8
# res2 = 20
# res3 = 12
# res4 = 14

# # Band definitions
# # Center frequency and BW taken from OLIMPO
# Bands_list_five_chan = [{'name':'Band 1','nu_meanGHz':145,'FBW':0.3,'nu_resGHz':res1,'NEP_aWrtHz':NEP1,'N_pixels':pix1},\
#       {'name':'Band 2','nu_meanGHz':250,'FBW':0.4,'nu_resGHz':res2,'NEP_aWrtHz':NEP2,'N_pixels':pix2},\
#       {'name':'Band 3','nu_meanGHz':365,'FBW':0.18,'nu_resGHz':res3,'NEP_aWrtHz':NEP3,'N_pixels':pix3},\
#       {'name':'Band 4','nu_meanGHz':460,'FBW':0.15,'nu_resGHz':res4,'NEP_aWrtHz':NEP4,'N_pixels':pix4}]

# # Band resolutions
# # Hybrid 3-5 channel
# res3 = 26
# res4 = 14

# # Band definitions
# # Center frequency and BW taken from OLIMPO
# Bands_list_three_five_chan = [{'name':'Band 1','nu_meanGHz':145,'FBW':0.3,'nu_resGHz':res1,'NEP_aWrtHz':NEP1,'N_pixels':pix1},\
#       {'name':'Band 2','nu_meanGHz':250,'FBW':0.4,'nu_resGHz':res2,'NEP_aWrtHz':NEP2,'N_pixels':pix2},\
#       {'name':'Band 3','nu_meanGHz':365,'FBW':0.18,'nu_resGHz':res3,'NEP_aWrtHz':NEP3,'N_pixels':pix3},\
#       {'name':'Band 4','nu_meanGHz':460,'FBW':0.15,'nu_resGHz':res4,'NEP_aWrtHz':NEP4,'N_pixels':pix4}]

# Pixels per band
# 11335 channel case
pix1 = 150
pix2 = 250
pix3 = 1000
pix4 = 1250
pix5 = 750

# Band resolutions
# Hybrid 3-3-5 First choice
# Band resolutions for SOFTS
res1 = 17
res2 = 39
res3 = 26
res4 = 27
res5 = 40#24

# olimpo_frequencies = np.asarray([145,250,365,460])
# #Calculated rms for 80 hour case
# olimpo_rms = np.asarray([0.36, 0.27, 0.70, 1.76])

# Band definitions
# Center frequency and BW taken from OLIMPO
Bands_list_hybrid_first_band_third = [{'name':'Band 1','nu_meanGHz':145,'rms':0.36,'type':'OLIMPO'},\
      {'name':'Band 2','nu_meanGHz':250,'rms':0.36,'type':'OLIMPO'},\
      {'name':'Band 3','nu_meanGHz':365,'FBW':0.18,'nu_resGHz':res3,'N_pixels':pix3,'type':'spectrometric'},\
      {'name':'Band 4','nu_meanGHz':460,'FBW':0.15,'nu_resGHz':res4,'N_pixels':pix4,'type':'spectrometric'},\
      {'name':'Band 5','nu_meanGHz':660,'FBW':0.166,'nu_resGHz':res5,'N_pixels':pix5,'type':'spectrometric'}]

# res5 = 24
    
# # Band definitions
# # Center frequency and BW taken from OLIMPO
# Bands_list_hybrid_first_band_fifth = [{'name':'Band 1','nu_meanGHz':145,'FBW':0.3,'nu_resGHz':res1,'N_pixels':pix1},\
#       {'name':'Band 2','nu_meanGHz':250,'FBW':0.4,'nu_resGHz':res2,'N_pixels':pix2},\
#       {'name':'Band 3','nu_meanGHz':365,'FBW':0.18,'nu_resGHz':res3,'N_pixels':pix3},\
#       {'name':'Band 4','nu_meanGHz':460,'FBW':0.15,'nu_resGHz':res4,'N_pixels':pix4},\
#       {'name':'Band 5','nu_meanGHz':660,'FBW':0.166,'nu_resGHz':res5,'N_pixels':pix5}]

# res5 = 100
    
# # Band definitions
# # Center frequency and BW taken from OLIMPO
# Bands_list_hybrid_first_band_single = [{'name':'Band 1','nu_meanGHz':145,'FBW':0.3,'nu_resGHz':res1,'N_pixels':pix1},\
#       {'name':'Band 2','nu_meanGHz':250,'FBW':0.4,'nu_resGHz':res2,'N_pixels':pix2},\
#       {'name':'Band 3','nu_meanGHz':365,'FBW':0.18,'nu_resGHz':res3,'N_pixels':pix3},\
#       {'name':'Band 4','nu_meanGHz':460,'FBW':0.15,'nu_resGHz':res4,'N_pixels':pix4},\
#       {'name':'Band 5','nu_meanGHz':660,'FBW':0.166,'nu_resGHz':res5,'N_pixels':pix5}]
    
# # Band definitions
# # Center frequency and BW taken from OLIMPO
# Bands_list_hybrid_first_band_nofifth = [{'name':'Band 1','nu_meanGHz':145,'FBW':0.3,'nu_resGHz':res1,'N_pixels':pix1},\
#       {'name':'Band 2','nu_meanGHz':250,'FBW':0.4,'nu_resGHz':res2,'N_pixels':pix2},\
#       {'name':'Band 3','nu_meanGHz':365,'FBW':0.18,'nu_resGHz':res3,'N_pixels':pix3},\
#       {'name':'Band 4','nu_meanGHz':460,'FBW':0.15,'nu_resGHz':res4,'N_pixels':pix4}]

# # Band resolutions
# # Hybrid 3-3-5 
# # Band resolutions for SOFTS
# res1 = 17
# res2 = 39
# res3 = 26
# res4 = 27
# res5 = 30#22

# Bands_list_hybrid_second_band = [{'name':'Band 1','nu_meanGHz':145,'FBW':0.3,'nu_resGHz':res1,'N_pixels':pix1},\
#       {'name':'Band 2','nu_meanGHz':250,'FBW':0.4,'nu_resGHz':res2,'N_pixels':pix2},\
#       {'name':'Band 3','nu_meanGHz':365,'FBW':0.18,'nu_resGHz':res3,'N_pixels':pix3},\
#       {'name':'Band 4','nu_meanGHz':460,'FBW':0.15,'nu_resGHz':res4,'N_pixels':pix4},\
#       {'name':'Band 5','nu_meanGHz':890,'FBW':0.112,'nu_resGHz':res5,'N_pixels':pix5}]

# List of cluster y-values to explore
y = [15.0E-06]
     
# List of cluster electron temperatures to explore
electron_temperature = [5.0] #KeV   
     
# Constant fiducial velocity of 0
peculiar_vel = 1e-11 #km/s
betac = [peculiar_vel/(3e5)]
     
# Run names to save to
# run_names_hybrid_one = ["test5000_single_three"]
# run_names_hybrid_two = ["test5000_nofifth_three"]
# run_names_hybrid_three = ["test5000_third_three"]
# run_names_hybrid_four = ["test5000_fifth_three"]

run_name = ['test_123']

# Integration times for each run
times = [288000] #Seconds
     
# Loop over runs, define number of processors to use
for i in range(1):
#     biasclass = BIAS.Spectral_Simulation_SOFTS(y[i], electron_temperature[i], peculiar_vel, Bands_list_three_chan, times[i])
#     biasclass.run_sim(run_names_phot[i], "OLIMPO", processors_pool = 30)
    
#     biasclass = BIAS.Spectral_Simulation_SOFTS(y[i], electron_temperature[i], peculiar_vel, Bands_list_three_chan, times[i])
#     biasclass.run_sim(run_names_three[i], "SOFTS", processors_pool = 20)
    
#     biasclass = BIAS.Spectral_Simulation_SOFTS(y[i], electron_temperature[i], peculiar_vel, Bands_list_five_chan, times[i])
#     biasclass.run_sim(run_names_five[i], "SOFTS", processors_pool = 20)
    
#     biasclass = BIAS.Spectral_Simulation_SOFTS(y[i], electron_temperature[i], peculiar_vel, Bands_list_hybrid_first_band_single, times[i])
#     biasclass.run_sim(run_names_hybrid_one[i], "HYBRID", processors_pool = 30)
    
#     biasclass = BIAS.Spectral_Simulation_SOFTS(y[i], electron_temperature[i], peculiar_vel, Bands_list_hybrid_first_band_nofifth, times[i])
#     biasclass.run_sim(run_names_hybrid_two[i], "HYBRID", processors_pool = 30)

#     biasclass = BIAS.Spectral_Simulation_SOFTS(y[i], electron_temperature[i], peculiar_vel, Bands_list_hybrid_first_band_third, times[i])
#     biasclass.run_sim(run_names_hybrid_three[i], "HYBRID", processors_pool = 30)
    
#     biasclass = BIAS.Spectral_Simulation_SOFTS(y[i], electron_temperature[i], peculiar_vel, Bands_list_hybrid_first_band_fifth, times[i])
#     biasclass.run_sim(run_names_hybrid_four[i], "HYBRID", processors_pool = 30)

      siftclass = SIFT.Spectral_Simulation(y[i], electron_temperature[i], peculiar_vel, Bands_list_hybrid_first_band_third, times[i])
      siftclass.run_sim(run_name[i], processors_pool = 30)
                                       
    
    
    

