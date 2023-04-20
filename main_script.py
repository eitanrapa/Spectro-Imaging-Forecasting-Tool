import BIAS_classes as BIAS
import numpy as np

#Band definitions
#3 Channel case
pix1 = 50
pix2 = 136
pix3 = 282
pix4 = 460

# I tune these by eye so that we get 3 channels per sub-band
res1 = 17
res2 = 39
res3 = 26
res4 = 27

NEP1 = np.sqrt(8.87**2 + 3.1**2)
NEP2 = np.sqrt(10.4**2 + 2.53**2)
NEP3 = np.sqrt(6.9**2 + 1.37**2)
NEP4 = np.sqrt(7.12**2 + 1.00**2) # all in aW/rtHz units

Bands_list1 = [{'name':'Band 1','nu_meanGHz':145,'FBW':0.3,'nu_resGHz':res1,'NEP_aWrtHz':NEP1,'N_pixels':pix1},\
      {'name':'Band 2','nu_meanGHz':250,'FBW':0.4,'nu_resGHz':res2,'NEP_aWrtHz':NEP2,'N_pixels':pix2},\
      {'name':'Band 3','nu_meanGHz':365,'FBW':0.18,'nu_resGHz':res3,'NEP_aWrtHz':NEP3,'N_pixels':pix3},\
      {'name':'Band 4','nu_meanGHz':460,'FBW':0.15,'nu_resGHz':res4,'NEP_aWrtHz':NEP4,'N_pixels':pix4}]

#Galaxy cluster parameters
y = [1.8e-5,6.0e-6,2.0e-6,6.0e-6,2.0e-6,5.4e-5,6.0e-6,2.0e-6,6.0e-6,2.0e-6]
     
electron_temperature = [5.0,5.0,0.5,0.5,0.5,5.0,5.0,5.0,0.5,0.5] #KeV   
     
peculiar_vel = 0.000001 #km/s
betac = [peculiar_vel/(3e5)]
     
run_names = ["306","307","310","311","312","319","321","322","323","324"]
     
times = [1.152e6,1.152e6,288000,1.152e6,1.152e6,72000,72000,72000,72000,72000]
     
for i in range(14):
    biasclass = BIAS.Spectral_Simulation_SOFTS(y[i], electron_temperature[i], peculiar_vel, Bands_list1, times[i])
    biasclass.run_sim(run_names[i], "HYBRID", processors_pool = 12)