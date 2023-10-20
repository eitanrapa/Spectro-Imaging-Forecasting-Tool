import SIFT_classes as SIFT
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import corner
import pygtc
import numpy as np
import pandas as pd

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

#Function to plot the spectra and OLIMPO, SOFTS bands
def Differential_Intensity_Projection(y_value, electron_temperature, peculiar_velocity, sides_long, sides_lat, bands, time):
    labels = ('tau','temperature','peculiar_velocity','a_sides','b_sides')
    freq = np.linspace(80e9,1000e9,2000)
    
    sz_template = SIFT.szpack_signal(freq, SIFT.y_to_tau(y_value, electron_temperature), electron_temperature, peculiar_velocity)
    
    sides_template = SIFT.SIDES_continuum(freq, sides_long, sides_lat)
    
    plt.rc('xtick',labelsize=18)
    plt.rc('ytick',labelsize=18)
    fig = plt.figure(figsize=(10,10))
  
    #Plot SZ components
    plt.plot(freq*HztoGHz,abs(sz_template),'--k',label='Total SZ',linewidth=2)
    plt.plot(freq*HztoGHz,abs(SIFT.szpack_signal(freq, SIFT.y_to_tau(y_value, electron_temperature), electron_temperature, 1e-11) - SIFT.classical_tsz(y_value, freq)), label='rSZ ' + str(electron_temperature) + ' keV')
        
    plt.plot(freq*HztoGHz,abs(SIFT.classical_tsz(y_value,freq)),label='tSZ y=' + str(y_value))

    #SIDES continuum model and emission line data
    plt.plot(freq*HztoGHz,abs(sides_template),color='pink',label='SIDES continuum')
    
        
    nu_total_array = np.empty(0)
    total_sz_array = np.empty(0)
    sigma_b_array = np.empty(0)
    
    for band in bands:
        if (band['type'] == 'OLIMPO'):
            #80 hour normalized
            rms_value = band['rms']*(np.sqrt(80)/(np.sqrt(time/3600)))
            nu_vec_b = band['nu_meanGHz']*GHztoHz
            x = h_p*nu_vec_b/(k_b*TCMB)
            sigma_B_b = 2*k_b*((nu_vec_b/c)**2)*(x/(np.exp(x)-1))*(x*np.exp(x))/(np.exp(x)-1)*band['rms']*1e-6
            nu_total_array = np.concatenate((nu_total_array, nu_vec_b), axis=None)
            sigma_b_array = np.concatenate((sigma_b_array, sigma_B_b), axis=None)
        if (band['type'] == 'spectrometric'):
            nu_vec_b, sigma_B_b = SIFT.sigB(band, time)
            nu_total_array = np.concatenate((nu_total_array, nu_vec_b), axis=None)
            sigma_b_array = np.concatenate((sigma_b_array, sigma_B_b), axis=None)
             
    plt.plot(nu_total_array*HztoGHz, sigma_b_array,'o', lw=7, alpha=1, color='maroon')
        
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('GHz',fontsize=20)
    plt.ylabel('W/m^2/Hz/Sr',fontsize=20)

    #Make xticks match as best as possible
    plt.xticks(np.rint(np.logspace(np.log10(80), np.log10(1e3),num=9)),np.rint(np.logspace(np.log10(80), np.log10(1e3),num=9)))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, prop={'size':12}, ncol=1, title= '{} hour obs.'.format(time/3600))
    plt.show()
    return None
    
# Function to plot contour plots of run
def Contour_Plot_Projection(run_number, y_value, electron_temperature, peculiar_velocity, a_sides=1, b_sides=1):
        
    # Read simulation output, change directory 
    df = pd.read_csv('/bolocam/bolocam/erapaport/runs/mcmc_run_{}'.format(run_number),header=None) 
    data = df.to_numpy()
        
    # Chain length - discard number = 8000
       
#         bad_indices = []#[0,3,7,10,17,18,24,28,28,23] #manual modification discarding bright points, dependent on parameter_file
#         new_data = data
#         for i in range(len(bad_indices)):
#             new_data = np.concatenate((new_data[:8000*(bad_indices[i]),:],new_data[8000*((bad_indices[i])+1):,:]),axis=0)
            
    labels = ('y','temperature','peculiar_velocity','a_sides','b_sides')
    theta = (y_value, electron_temperature, peculiar_velocity, a_sides, b_sides)

    #Plot contour plot
    fig = corner.corner(
    data, labels=labels, truths=theta, smooth = 0
    );
        
    return fig, data
    
# Function to plot comparisons of 2 runs
def Contour_Plot_Double_Projection(run_number1, run_number2, y_value, electron_temperature, peculiar_velocity, a_sides=1, b_sides=1):
    
    # Read simulation output, change directory 
    df1 = pd.read_csv('/bolocam/bolocam/erapaport/runs/mcmc_run_{}'.format(run_number1),header=None) 
    data1 = df1.to_numpy()
        
    df2 = pd.read_csv('/bolocam/bolocam/erapaport/runs/mcmc_run_{}'.format(run_number2),header=None) 
    data2 = df2.to_numpy()
        
    chainLabels = ["Run {}".format(run_number1),
           "Run {}".format(run_number2)]
        
#     bad_indices = [0,3,7,10,17,18,24,28,28,23] #manual modification discarding bright points, dependent on parameter_file

    # Chain length - discard number = 8000
        
#     for i in range(len(bad_indices)):
#         new_data1 = np.concatenate((new_data1[:8000*(bad_indices[i]),:],new_data1[8000*((bad_indices[i])+1):,:]),axis=0)
            
#     for i in range(len(bad_indices)):
#         new_data2 = np.concatenate((new_data2[:8000*(bad_indices[i]),:],new_data2[8000*((bad_indices[i])+1):,:]),axis=0)

    labels = ('y','temperature','peculiar_vel','a_sides','b_sides')
    theta = (y_value, electron_temperature, peculiar_velocity, a_sides, b_sides)

    #Plot contour plot
    GTC = pygtc.plotGTC(chains=[data1,data2], chainLabels=chainLabels, truths=theta, paramNames = labels,figureSize=10)
        
    return GTC, data1, data2

def Chain_Projection(run_number, y_value, electron_temperature, peculiar_velocity, a_sides=1, b_sides=1):
        
    # Read simulation output, change directory 
    df = pd.read_csv('/bolocam/bolocam/erapaport/runs/mcmc_run_{}'.format(run_number),header=None) 
    data = df.to_numpy()
        
    labels = ('y','temperature','peculiar_velocity','a_sides','b_sides')
    theta = (y_value, electron_temperature, peculiar_velocity, a_sides, b_sides)
        
    fig, axes = plt.subplots(5, figsize=(30,40), sharex=True)
    ndim = 5
    for i in range(ndim):
        ax = axes[i]
        ax.plot(data[:,i], "k", alpha=0.3)
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
            
    return fig, axes