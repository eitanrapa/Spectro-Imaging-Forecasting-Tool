import MCMC_SOFTS_A2A_ICM as MCSZ
import numpy as np
import matplotlib.pyplot as plt
import time
from astropy import units as u
import csv
import pandas as pd
from astropy.io import fits
import sys

y_val = float(sys.argv[1])
Time_val = float(sys.argv[2])
run_number = int(sys.argv[3])

#Import CMB data
fname = '/data/bolocam/bolocam/erapaport/cmb.fits'
hdu = fits.open(fname)
image_data = hdu[1].data

#Specify parameter values

#Galaxy cluster parameters
#y = 6e-6
y = y_val
electron_temperature = 5.0 #KeV
tau = MCSZ.y_to_tau(y,electron_temperature)
peculiar_vel = 0.000001 #km/s
betac = peculiar_vel/(3e5)

#SIDES model values
amp_sides = 1
b_sides = 1

#Define frequencies and rms

#Time, Timelabel = 86400, '1 day' #1 day
#Time, Timelabel = 604800, '1 week'  #1 week
#Time, Timelabel = 3600.0, '1 hour'  #1 hour
#Time, Timelabel = 3.154E7, '1 year'  # 1 year

#Time, Timelabel = 288000, '80 hours'
#Time, Timelabel = 72000, '20 hours'
Time = Time_val

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

#MCMC hyperparameters

chain_length = 10000
walkers = 100
realizations = 40
discard_n = 2000
thin_n = 100
processors_pool = 30

#Read previous parameters if needed
df = pd.read_csv('/data/bolocam/bolocam/erapaport/mcmc_run009_parameters',header=None) 
params = df.to_numpy()

# # Comment out if using previous parameters
# parameter_matrix = np.empty((realizations,8))

samples = [[0,0,0,0,0]]
samples = np.asarray(samples)

for i in range(realizations):

#     #Random cmb anis. value
#     cmb_long = np.random.randint(2,197)
#     cmb_lat = np.random.randint(2,197)
#     amp_cmb = image_data[cmb_long][cmb_lat] - np.mean(image_data[cmb_long-2:cmb_long+3,cmb_lat-2:cmb_lat+3])

#     #Random SIDES pixel
#     sides_long = np.random.randint(0,154)
#     sides_lat = np.random.randint(0,154)

#     #Random cmb ksz,tsz anis. values
#     amp_ksz = np.random.normal(0,6e-7)
#     amp_tsz = np.random.normal(0,6e-7)
#     parameter_matrix[i] = [y, electron_temperature, betac, sides_long, sides_lat, amp_cmb, amp_ksz, amp_tsz]
    
    #Uncomment if using previous parameters
    amp_cmb = params[i,5]
    sides_long = int(params[i,3])
    sides_lat = int(params[i,4])
    amp_ksz = params[i,6]
    amp_tsz= params[i,7]
    
    theta = (y, electron_temperature, betac, amp_sides, b_sides)
    anisotropies = (amp_ksz, amp_tsz, amp_cmb)
    start_time = time.time()
    sampler = MCSZ.mcmc(theta, anisotropies,  Bands_list1, Time, sides_long, sides_lat, chain_length, walkers, processors_pool)
    print("--- %s seconds ---" % (time.time() - start_time))
    samples = np.append(samples,sampler.get_chain(discard=discard_n, flat=True, thin=thin_n),axis=0)
    
#Write simulation output, change directory/name
with open('/data/bolocam/bolocam/erapaport/mcmc_run_{}'.format(run_number), 'w') as f:
    write = csv.writer(f)
    write.writerows(samples)
    
# #Write parameters if needed, change directory/name
# with open('/data/bolocam/bolocam/erapaport/mcmc_run012_parameters', 'w') as f:
#     write = csv.writer(f)
#     write.writerows(parameter_matrix)
