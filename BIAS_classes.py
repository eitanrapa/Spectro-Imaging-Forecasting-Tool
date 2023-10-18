import numpy as np
import math
import sys, os
# Make sure to append SZpack path
sys.path.append("/home/bolocam/erapaport/Spectro-Imaging-Forecasting-Tool/codes/SZpack.v1.1.1/python")
sys.path.append("/home/bolocam/erapaport/BIAS/Auxiliary")
import Mather_photonNEP12a as NEP
import pandas as pd
import csv
import SZpack as sz
import emcee
from astropy.io import fits
from scipy.interpolate import interp1d
from multiprocessing import Pool
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import corner
import pygtc

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

# #SOFTS function
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

#SOFTS function
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
Define classes for SOFTS, OLIMPO, Hybrid Sims
"""

class Spectral_Simulation_SOFTS:
    
    def __init__(self, y_value, electron_temperature, peculiar_velocity, bands, time,  amp_sides = 1, b_sides = 1):
        self.y_value = y_value
        self.electron_temperature = electron_temperature
        self.peculiar_velocity = peculiar_velocity
        self.bands = bands
        self.time = time
        self.amp_sides = amp_sides
        self.b_sides = b_sides
    
    #Model used for MCMC calculation
    def model(self, theta, anisotropies, freq):
        y, temperature, peculiar_velocity, amp_sides, b_sides = theta
        ksz_anis, tsz_anis, cmb_anis = anisotropies
        
        #Read SIDES average model
        df = pd.read_csv('/data/bolocam/bolocam/erapaport/sides.csv',header=None) 
        data = df.to_numpy()
        data = data.squeeze()
        SIDES = data*MJyperSrtoSI
    
        #Modify template by amp_sides, b_sides 
        sides_template = amp_sides*interpolate(freq,SIDES,np.linspace(0,1500e9,751)*b_sides)
    
        #CMB and galaxy cluster SZ template
        cmb_template = dB(cmb_anis + ksz_anis,freq)
        sz_template = szpack_signal(freq, y_to_tau(y + tsz_anis,temperature), temperature, peculiar_velocity)
    
        template_total = sz_template + sides_template + cmb_template
        return template_total
    
    #Individual templates for plotting using SIDES fits data
    def templates(self, freq, long, lang):
    
        #Read FITS file
        fname = '/data/bolocam/bolocam/erapaport/continuum.fits'
        hdu = fits.open(fname)
        image_data = hdu[0].data
    
        total_SIDES = np.zeros(751)
        #Rebinning for 3 arcmin resolution
        for col in range(6):
            for row in range(6):
                total_SIDES += image_data[:,long + row,lang + col]*MJyperSrtoSI
        total_SIDES = total_SIDES/36
    
        #Interpolate for specific frequencies
        sides_template = interpolate(freq,total_SIDES,np.linspace(0,1500e9,751))
    
        #Galaxy cluster SZ template
        sz_template = szpack_signal(freq, y_to_tau(self.y_value,self.electron_temperature), self.electron_temperature, self.peculiar_velocity)
    
        template_total = [sz_template,sides_template]
        return template_total
    
    def log_likelihood(self, theta, anisotropies, freq, data, noise):
        modeldata = self.model(theta, anisotropies, freq)
        return -0.5 * np.sum(((data - modeldata)/noise)**2)
    
    #Change priors if needed
    def log_prior(self, theta):
        y, temperature, peculiar_velocity, amp_sides, b_sides = theta
    
        if (y < 0 or y > 0.1):
            return -np.inf
        if ((peculiar_velocity/3e5) < -0.02 or (peculiar_velocity/3e5) > 0.02):
            return -np.inf
        #if (temperature < 2.73 or temperature > 20):
        #    return -np.inf
        if (amp_sides < 0 or amp_sides > 2.5):
            return -np.inf
        if (b_sides < 0 or b_sides > 2.5):
            return -np.inf
        mu = self.electron_temperature
        sigma = self.electron_temperature/10 #10% precision
        
        #Gaussian prior on temperature with mean of mu and std. deviation of sigma
        return np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(temperature-mu)**2/sigma**2
    
    def log_probability(self, theta, anisotropies,freq, data, noise):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta, anisotropies, freq, data, noise)  
    
    def mcmc(self, run_type, anisotropies, long, lang,  max_n, walkern, processors):
        ksz_anis, tsz_anis, cmb_anis = anisotropies
    
        nu_total_array = np.empty(0)
        total_sz_array = np.empty(0)
        sigma_b_array = np.empty(0)
    
        if (run_type == "SOFTS"):
            #Create list of frequencies and NESB 
            #Use SOFTS function to calculate Sig b
            for bb in range(len(self.bands)):
                b = self.bands[bb]
                nu_vec_b, sigma_B_b = sigB(b, self.time)
                nu_total_array = np.concatenate((nu_total_array, nu_vec_b),axis=None)
                sigma_b_array = np.concatenate((sigma_b_array, sigma_B_b),axis=None)
       
        if (run_type == "OLIMPO"):
            #Use RMS values from calculation of 80 hour case using OLIMPO definitions
            rms_values = np.asarray([0.36, 0.27, 0.70, 1.76])*(np.sqrt(80)/(np.sqrt(self.time/3600)))
            frequencies = [145,250,365,460]
            nu_total_array = np.array(frequencies)*GHztoHz
            x = h_p*nu_total_array/(k_b*TCMB)
            sigma_b_array = 2*k_b*((nu_total_array/c)**2)*(x/(np.exp(x)-1))*(x*np.exp(x))/(np.exp(x)-1)*np.array(rms_values)*1e-6
    
        if (run_type == "HYBRID"):
            #Take 2 first channels as OLIMPO channels, and then the rest as SOFTS channels
            rms_values = np.asarray([0.36, 0.27])*(np.sqrt(80)/(np.sqrt(self.time/3600)))
            frequencies = [145,250]
            nu_total_array = np.array(frequencies)*GHztoHz
            x = h_p*nu_total_array/(k_b*TCMB)
            sigma_b_array = 2*k_b*((nu_total_array/c)**2)*(x/(np.exp(x)-1))*(x*np.exp(x))/(np.exp(x)-1)*np.array(rms_values)*1e-6
            for bb in range(len(self.bands)-2):
                b = self.bands[bb+2]
                nu_vec_b, sigma_B_b = sigB(b, self.time)
                nu_total_array = np.concatenate((nu_total_array, nu_vec_b),axis=None)
                sigma_b_array = np.concatenate((sigma_b_array, sigma_B_b),axis=None)
                
        #Read FITS file
        fname = '/data/bolocam/bolocam/erapaport/continuum.fits'
        hdu = fits.open(fname)
        image_data = hdu[0].data
    
        total_SIDES = np.zeros(751)
        #Rebinning for 3.0 arcmin resolution
        for col in range(6):
            for row in range(6):
                total_SIDES += image_data[:,long + row,lang + col]*MJyperSrtoSI
        total_SIDES = total_SIDES/36
    
        #Interpolate for specific frequencies
        sides_template = interpolate(nu_total_array,total_SIDES,np.linspace(0,1500e9,751))
    
        #CMB and galaxy cluster SZ template
        sz_template = szpack_signal(nu_total_array,y_to_tau(self.y_value + tsz_anis, self.electron_temperature), self.electron_temperature, self.peculiar_velocity)
        cmb_template = dB(cmb_anis + ksz_anis, nu_total_array)
        total_sz_array = sz_template + cmb_template + sides_template
        
        theta = self.y_value, self.electron_temperature, self.peculiar_velocity, self.amp_sides, self.b_sides
        pos = []
        for item in theta:
            pos.append(item*(1 + 0.01*np.random.randn(walkern)))
        pos_array= np.asarray(pos)
        pos_array = pos_array.transpose()
        nwalkers, ndim = pos_array.shape

        with Pool(processors) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability, args=(anisotropies, nu_total_array, total_sz_array, sigma_b_array),pool=pool)
            for sample in sampler.sample(pos_array, iterations=max_n, progress=True):
                continue
        
        return sampler
    
    def run_sim(self, run_number, run_type, chain_length = 15000, walkers = 200, realizations = 1, discard_n = 7000, thin_n = 100, processors_pool = 30):   
        
        #Read saved parameters file
        df = pd.read_csv('/data/bolocam/bolocam/erapaport/parameter_file',header=None) 
        params = df.to_numpy()

        samples = [[0,0,0,0,0]]
        samples = np.asarray(samples)

        for i in range(realizations):
            i = 2
            amp_cmb = params[i,5]
            sides_long = int(params[i,3])
            sides_lat = int(params[i,4])
            amp_ksz = params[i,6]
            amp_tsz= params[i,7]
            anisotropies = (amp_ksz, amp_tsz, amp_cmb)
            sampler = self.mcmc(run_type, anisotropies, sides_long, sides_lat, chain_length, walkers, processors_pool)
            samples = np.append(samples,sampler.get_chain(discard=discard_n, flat=True, thin=thin_n),axis=0)
           
        #Write simulation output, change directory/name
        with open('/data/bolocam/bolocam/erapaport/mcmc_run_{}'.format(run_number), 'w') as f:
            write = csv.writer(f)
            write.writerows(samples)
    
    #Function to plot the spectra and OLIMPO, SOFTS bands
    def Differential_Intensity_Projection(self, run_type, long, lat):
        labels = ('tau','temperature','peculiar_velocity','amp_sides','b_sides')
        #freq = np.linspace(80e9,720e9,1000)
        freq = np.linspace(80e9,1000e9,1000)
        templates_complete = self.templates(freq, long, lat)
  
        plt.rc('xtick',labelsize=18)
        plt.rc('ytick',labelsize=18)
        fig = plt.figure(figsize=(10,10))
  
        #Plot SZ components
        plt.plot(freq*HztoGHz,abs(templates_complete[0]),'--k',label='Total SZ',linewidth=2)
        plt.plot(freq*HztoGHz,abs(szpack_signal(freq,y_to_tau(self.y_value,self.electron_temperature),self.electron_temperature,1e-11) - classical_tsz(self.y_value,freq)),label='rSZ ' + str(self.electron_temperature) + ' keV')
        
        plt.plot(freq*HztoGHz,abs(classical_tsz(self.y_value,freq)),label='tSZ y=' + str(self.y_value))

        #SIDES continuum model and emission line data
        plt.plot(freq*HztoGHz,abs(templates_complete[1]),color='pink',label='SIDES continuum')
        
        if (run_type == "SOFTS"):
            ncolor = len(self.bands)
            color=iter(cm.gray_r(np.linspace(1,0.8,ncolor+1)))
            for bb in range(ncolor):
                b1 = self.bands[bb]
                nu_vec_b1, sigma_B_b1 = sigB(b1, self.time)

                colr=next(color)
                plt.plot(nu_vec_b1*HztoGHz, sigma_B_b1,'o', lw=7, alpha=1, color='maroon')
            
        if (run_type == "OLIMPO"):
            rms_values = np.asarray([0.36, 0.27, 0.70, 1.76])*(np.sqrt(80)/(np.sqrt(self.time/3600)))
            frequencies = [145,250,365,460]
            nu_total_array = np.array(frequencies)*GHztoHz
            x = h_p*nu_total_array/(k_b*TCMB)
            sigma_b_array = 2*k_b*((nu_total_array/c)**2)*(x/(np.exp(x)-1))*(x*np.exp(x))/(np.exp(x)-1)*np.array(rms_values)*1e-6
            plt.scatter(nu_total_array*HztoGHz,sigma_b_array)
            
        if (run_type == "HYBRID"):
            rms_values = np.asarray([0.36, 0.27])*(np.sqrt(80)/(np.sqrt(self.time/3600)))
            frequencies = [145,250]
            nu_total_array = np.array(frequencies)*GHztoHz
            x = h_p*nu_total_array/(k_b*TCMB)
            sigma_b_array = 2*k_b*((nu_total_array/c)**2)*(x/(np.exp(x)-1))*(x*np.exp(x))/(np.exp(x)-1)*np.array(rms_values)*1e-6
            plt.scatter(nu_total_array*HztoGHz,sigma_b_array)
            ncolor = len(self.bands)-2
            color=iter(cm.gray_r(np.linspace(1,0.8,ncolor+1)))
            for bb in range(ncolor):
                b1 = self.bands[bb+2]
                nu_vec_b1, sigma_B_b1 = sigB(b1, self.time)

                colr=next(color)
                plt.plot(nu_vec_b1*HztoGHz, sigma_B_b1,'o', lw=7, alpha=1, color='maroon')
            
            
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('GHz',fontsize=20)
        plt.ylabel('W/m^2/Hz/Sr',fontsize=20)

        #Make xticks match as best as possible
        plt.xticks(np.rint(np.logspace(np.log10(80), np.log10(1e3),num=9)),np.rint(np.logspace(np.log10(80), np.log10(1e3),num=9)))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, prop={'size':12}, ncol=1, title= '{} hour obs.'.format(self.time/3600))
        fig.savefig('temp.png', transparent=True)
        plt.show()
        return None
    
    # Function to plot contour plots of run
    def Contour_Plot_Projection(self, run_number):
        
        # Read simulation output, change directory 
        df = pd.read_csv('/data/bolocam/bolocam/erapaport/mcmc_run_{}'.format(run_number),header=None) 
        data = df.to_numpy()
        
        # Chain length - discard number = 8000
        
        bad_indices = []#[0,3,7,10,17,18,24,28,28,23] #manual modification discarding bright points, dependent on parameter_file
        new_data = data[1:,:]
        for i in range(len(bad_indices)):
            new_data = np.concatenate((new_data[:8000*(bad_indices[i]),:],new_data[8000*((bad_indices[i])+1):,:]),axis=0)
            
        labels = ('y','temperature','peculiar_velocity','amp_sides','b_sides')
        theta = (self.y_value, self.electron_temperature, self.peculiar_velocity, self.amp_sides, self.b_sides)

        #Plot contour plot
        fig = corner.corner(
        new_data[:,:], labels=labels, truths=theta, smooth = 0
        );
        
        return fig, new_data
    
    # Function to plot comparisons of 2 runs
    def Contour_Plot_Double_Projection(self, run_number1, run_number2):
        
        # Read simulation output, change directory 
        df1 = pd.read_csv('/data/bolocam/bolocam/erapaport/mcmc_run_{}'.format(run_number1),header=None) 
        data1 = df1.to_numpy()
        
        df2 = pd.read_csv('/data/bolocam/bolocam/erapaport/mcmc_run_{}'.format(run_number2),header=None) 
        data2 = df2.to_numpy()
        
        chainLabels = ["Run {}".format(run_number1),
               "Run {}".format(run_number2)]
        
        bad_indices = [0,3,7,10,17,18,24,28,28,23] #manual modification discarding bright points, dependent on parameter_file
        
        new_data1 = data1[1:,:]
        new_data2 = data2[1:,:]

        # Chain length - discard number = 8000
        
        for i in range(len(bad_indices)):
            new_data1 = np.concatenate((new_data1[:8000*(bad_indices[i]),:],new_data1[8000*((bad_indices[i])+1):,:]),axis=0)
            
        for i in range(len(bad_indices)):
            new_data2 = np.concatenate((new_data2[:8000*(bad_indices[i]),:],new_data2[8000*((bad_indices[i])+1):,:]),axis=0)

        labels = ('y','temperature','peculiar_vel','amp_sides','b_sides')
        theta = (self.y_value, self.electron_temperature, self.peculiar_velocity, self.amp_sides, self.b_sides)

        #Plot contour plot
        GTC = pygtc.plotGTC(chains=[new_data1,new_data2], chainLabels=chainLabels, truths=theta, paramNames = labels,figureSize=10)
        
        return GTC, new_data1, new_data2
