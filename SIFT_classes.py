import numpy as np
import math
import sys, os
# Make sure to append SZpack path
sys.path.append("/home/bolocam/erapaport/SIFT_old/codes/SZpack.v1.1.1/python")
sys.path.append("/bolocam/bolocam/erapaport/Auxiliary/")
import Mather_photonNEP12a as NEP
import pandas as pd
import csv
import SZpack as sz
import emcee
from astropy.io import fits
from scipy.interpolate import interp1d
from multiprocessing import Pool

'''
Cosmological terms from G. Sun
'''

c = 299792458.0                                         # Speed of light - [c] = m/s
h_p = 6.626068e-34                                      # Planck's constant in SI units
k_b = 1.38065e-23                                       # Boltzmann constant in SI units
MJyperSrtoSI = 1e-20                                    # MegaJansky/Sr to SI units
GHztoHz = 1e9                                           # Gigahertz to hertz
HztoGHz = 1e-9                                          # Hertz to Gigahertz
TCMB = 2.725                                            # Canonical CMB in Kelvin
m = 9.109*10**(-31)                                     # Electron mass in kgs

# #Noise Equivalent Brightness function with known NEPs
# def sigB(band_details, Time):
#     # Use for apples to apples with OLIMPO photo
    
#     BW_GHz = band_details['nu_meanGHz']*band_details['FBW']
    
#     nu_min = (band_details['nu_meanGHz'] - 0.5*BW_GHz)*GHztoHz
#     nu_max = (band_details['nu_meanGHz'] + 0.5*BW_GHz)*GHztoHz
#     nu_res = band_details['nu_resGHz']*GHztoHz
#     Npx = band_details['N_pixels']
    
#     NEP_tot = (band_details['NEP_aWrtHz'])*1E-18
#     Nse = int(np.round(BW_GHz/band_details['nu_resGHz']))
#     nu_vec = np.linspace(nu_min, nu_max, Nse)
#     AOnu = (c/nu_vec)**2
    
#     #Defined empirically to match OLIMPO inefficiencies at single channel bands
#     inefficiency = 0.019
#     delP = 2.0*NEP_tot/np.sqrt(Time*Npx)
#     sigma_B = delP/(AOnu)/nu_res/inefficiency

#     return nu_vec, sigma_B

#Noise Equivalent Brightness function with unknown NEPs
def sigB(band_details, Time, Tnoise=3.0):
    # Use for apples to apples with OLIMPO photo
    
    BW_GHz = band_details['nu_meanGHz']*band_details['FBW']
    
    nu_min = (band_details['nu_meanGHz'] - 0.5*BW_GHz)*GHztoHz
    nu_max = (band_details['nu_meanGHz'] + 0.5*BW_GHz)*GHztoHz
    nu_res = band_details['nu_resGHz']*GHztoHz
    Npx = band_details['N_pixels']
    
    NEP_phot1 = NEP.photonNEPdifflim(nu_min, nu_max, Tnoise) #This is CMB Tnoise
    NEP_phot2 = NEP.photonNEPdifflim(nu_min, nu_max, 10.0, aef=0.01) #Use real south pole data
    NEP_det = 10e-18 # ATTO WATTS per square-root(hz)
    NEP_tot = np.sqrt(NEP_phot1**2 + NEP_phot2**2 + NEP_det**2) #Dont include atmosphere for now
    # in making nu_vec we must be aware of resolution
    Nse = int(np.round(BW_GHz/band_details['nu_resGHz']))
    nu_vec = np.linspace(nu_min, nu_max, Nse)
    AOnu = (c/nu_vec)**2
    
    #Defined empirically to match OLIMPO inefficiencies at single channel bands
    inefficiency = 0.019
    delP = 2.0*NEP_tot/np.sqrt(Time*Npx)
    sigma_B = delP/(AOnu)/nu_res/inefficiency

    return nu_vec, sigma_B

#CMB Anisotropy function
def dB(dt, frequency):
    temp = TCMB/(1+dt)
    x = (h_p/(k_b * temp))*frequency
    I=((2*h_p)/(c**2))*(k_b*temp/h_p)**3
    return I*(x**4*np.exp(x)/((np.exp(x)-1)**2))*dt

#SZ total integration function
def szpack_signal(frequency, tau, temperature, peculiar_velocity):
    x_b = (h_p/(k_b * TCMB))*frequency
    original_x_b = (h_p/(k_b * TCMB))*frequency
    sz.compute_combo_means(x_b,tau,temperature,peculiar_velocity/3e5,0,0,0,0)
    return x_b*13.33914078*(TCMB**3)*(original_x_b**3)*MJyperSrtoSI

#Classical tSZ analytical calculation
def classical_tsz(y,frequency):
    x_b = (h_p/(k_b * TCMB))*frequency
    bv = 2*k_b*TCMB*((frequency**2)/(c**2))*(x_b/(np.exp(x_b)-1))
    return y*((x_b*np.exp(x_b))/(np.exp(x_b)-1))*(x_b*((np.exp(x_b)+1)/(np.exp(x_b)-1))-4)*bv

def SIDES_continuum(freq, long, lat):
            
    #Read FITS file
    fname = '/bolocam/bolocam/erapaport/Auxiliary/continuum.fits'
    hdu = fits.open(fname)
    image_data = hdu[0].data
    
    #SIDES spans 0 to 1500 GHz with 2 GHz intervals, with 0.5 arcmin resolution
    total_SIDES = np.zeros(751)
    
    #Rebinning for 3 arcmin
    for col in range(6):
        for row in range(6):
            total_SIDES += image_data[:, long + row, lat + col]*MJyperSrtoSI
    total_SIDES = total_SIDES/36
    
    #Interpolate for specific frequencies
    sides_template = interpolate(freq, total_SIDES, np.linspace(0,1500e9,751))
    return sides_template

def SIDES_average(freq, a_sides, b_sides):
    
    #Read SIDES average model
    df = pd.read_csv('/bolocam/bolocam/erapaport/Auxiliary/sides.csv',header=None) 
    data = df.to_numpy()
    data = data.squeeze()
    SIDES = data*MJyperSrtoSI
    
    #Modify template by a_sides, b_sides 
    sides_template = a_sides*interpolate(freq, SIDES, np.linspace(0,1500e9,751)*b_sides)
    return sides_template

#Optical depth to compton-y
def tau_to_y(tau, temperature):
    if (tau == 0):
        return 0
    return (tau*(temperature*11604525.0061598)*k_b) / (m*(c**2))

#Compton-y to optical depth
def y_to_tau(y, temperature):
    if (y == 0):
        return 0
    return (m*(c**2)*y) / (k_b*(temperature*11604525.0061598))

#Interpolation function
def interpolate(freq, datay, datax):
    f = interp1d(np.log(datax),np.log(datay),kind='slinear',bounds_error=False,fill_value=0)
    new_data = f(np.log(freq))
    return np.exp(new_data)

"""
Define class for simulation
"""

class Spectral_Simulation:
    
    def __init__(self, y_value, electron_temperature, peculiar_velocity, bands, time,  a_sides = 1, b_sides = 1):
        self.y_value = y_value
        self.electron_temperature = electron_temperature
        self.peculiar_velocity = peculiar_velocity
        self.bands = bands
        self.time = time
        self.a_sides = a_sides
        self.b_sides = b_sides
    
    #Model used for MCMC calculation
    def model(self, theta, freq):
        y, temperature, peculiar_velocity, a_sides, b_sides = theta
        
        #SIDES
        sides_template = SIDES_average(freq, a_sides, b_sides)

        sz_template = szpack_signal(freq, y_to_tau(y, temperature), temperature, peculiar_velocity)
    
        template_total = sz_template + sides_template
        return template_total
    
    #Individual templates for plotting using SIDES fits data
    def templates(self, freq, long, lat):
        
        sides_template = SIDES_continuum(freq, long, lat)
    
        #Galaxy cluster SZ template
        sz_template = szpack_signal(freq, y_to_tau(self.y_value, self.electron_temperature), self.electron_temperature, self.peculiar_velocity)
    
        template_total = [sz_template, sides_template]
        return template_total
    
    def log_likelihood(self, theta, freq, data, noise):
        modeldata = self.model(theta, freq)
        return -0.5 * np.sum(((data - modeldata)/noise)**2)
    
    #Change priors if needed
    def log_prior(self, theta):
        y, temperature, peculiar_velocity, a_sides, b_sides = theta
    
        if (y < 0 or y > 0.1):
            return -np.inf
        if ((peculiar_velocity/3e5) < -0.02 or (peculiar_velocity/3e5) > 0.02):
            return -np.inf
        if (temperature <  2.0  or temperature > 75.0):
            return -np.inf
        if (a_sides < 0 or a_sides > 2.5):
            return -np.inf
        if (b_sides < 0 or b_sides > 2.5):
            return -np.inf
        mu = self.electron_temperature
        sigma = self.electron_temperature/10 #10% precision
        
        #Gaussian prior on temperature with mean of mu and std. deviation of sigma
        return np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(temperature-mu)**2/sigma**2
    
    def log_probability(self, theta, freq, data, noise):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta, freq, data, noise)  
    
    def mcmc(self, anisotropies, long, lat, walkers, processors, chain_length):
        ksz_anis, tsz_anis, cmb_anis = anisotropies
    
        nu_total_array = np.empty(0)
        total_sz_array = np.empty(0)
        sigma_b_array = np.empty(0)
        
        for band in self.bands:
            if (band['type'] == 'OLIMPO'):
                #80 hour normalized
                rms_value = band['rms']*(np.sqrt(80)/(np.sqrt(self.time/3600)))
                nu_vec_b = band['nu_meanGHz']*GHztoHz
                x = h_p*nu_vec_b/(k_b*TCMB)
                sigma_B_b = 2*k_b*((nu_vec_b/c)**2)*(x/(np.exp(x)-1))*(x*np.exp(x))/(np.exp(x)-1)*band['rms']*1e-6
                nu_total_array = np.concatenate((nu_total_array, nu_vec_b), axis=None)
                sigma_b_array = np.concatenate((sigma_b_array, sigma_B_b), axis=None)
            if (band['type'] == 'spectrometric'):
                nu_vec_b, sigma_B_b = sigB(band, self.time)
                nu_total_array = np.concatenate((nu_total_array, nu_vec_b), axis=None)
                sigma_b_array = np.concatenate((sigma_b_array, sigma_B_b), axis=None)
        
        sides_template = SIDES_continuum(nu_total_array, long, lat)
    
        #CMB and galaxy cluster SZ template
        sz_template = szpack_signal(nu_total_array,y_to_tau(self.y_value, self.electron_temperature), self.electron_temperature, self.peculiar_velocity)
        tsz_template = classical_tsz(tsz_anis, nu_total_array)
        cmb_template = dB(cmb_anis, nu_total_array)
        ksz_template = dB(ksz_anis, nu_total_array)
        total_sz_array = sz_template + cmb_template + sides_template + ksz_anis + tsz_template
        
        theta = self.y_value, self.electron_temperature, self.peculiar_velocity, self.a_sides, self.b_sides
        pos = []
        for item in theta:
            pos.append(item*(1 + 0.01*np.random.randn(walkers)))
        pos_array= np.asarray(pos)
        pos_array = pos_array.transpose()
        nwalkers, ndim = pos_array.shape

        with Pool(processors) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability, args=(nu_total_array, total_sz_array, sigma_b_array),pool=pool)
            for sample in sampler.sample(pos_array, iterations=chain_length, progress=True):
                continue
        
        return sampler
    
    def run_sim(self, run_number, chain_length = 10000, walkers = 100, realizations = 100, discard_n = 2000, thin_n = 100, processors_pool = 30):   
        
        #Read saved parameters file
        df = pd.read_csv('/bolocam/bolocam/erapaport/Auxiliary/parameter_file_100',header=None) 
        params = df.to_numpy()

        samples = [[0,0,0,0,0]]
        samples = np.asarray(samples)

        for i in range(realizations):
            sides_long = int(params[i,0])
            sides_lat = int(params[i,1])
            amp_cmb = params[i,2]
            amp_ksz = params[i,3]
            amp_tsz= params[i,4]
            anisotropies = (amp_ksz, amp_tsz, amp_cmb)
            sampler = self.mcmc(anisotropies, sides_long, sides_lat, walkers, processors_pool, chain_length)
            samples = np.append(samples,sampler.get_chain(discard=discard_n, flat=True, thin=thin_n),axis=0)
            
        samples = samples[1:,:]
           
        #Write simulation output, change directory/name
        with open('/bolocam/bolocam/erapaport/new_runs/run_{}'.format(run_number), 'w') as f:
            write = csv.writer(f)
            write.writerows(samples)
