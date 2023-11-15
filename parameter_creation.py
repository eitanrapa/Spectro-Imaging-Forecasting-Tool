import csv
import numpy as np
from astropy.io import fits

#Read FITS file
fname = '/bolocam/bolocam/erapaport/Auxiliary/cmb_3arcmin.fits'
hdu = fits.open(fname)
image_data = hdu[1].data

params = [[0,0,0,0,0]]
params = np.asarray(params)
        
for i in range(100):
    sides_long = np.random.randint(0,160)
    sides_lat = np.random.randint(0,160)
    cmb_long = np.random.randint(2,197)
    cmb_lat = np.random.randint(2,197)
    amp_cmb = image_data[cmb_long][cmb_lat] - np.mean(image_data[cmb_long-2:cmb_long+3,cmb_lat-2:cmb_lat+3])
    amp_ksz = np.random.normal(0,6e-7)
    amp_tsz = np.random.normal(0,6e-7)
    params_real = [[sides_long, sides_lat, amp_cmb, amp_ksz, amp_tsz]]
    params = np.append(params, params_real,axis=0)
    
params = params[1:,:]

#Write simulation output, change directory/name
with open('/bolocam/bolocam/erapaport/Auxiliary/parameter_file_100', 'w') as f:
    write = csv.writer(f)
    write.writerows(params)