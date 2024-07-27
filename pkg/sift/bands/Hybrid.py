import numpy as np 
import Mather_photonNEP12a as NEP

# Constants
c = 299792458.0  # Speed of light - [c] = m/s
h_p = 6.626068e-34  # Planck's constant in SI units
k_b = 1.38065e-23  # Boltzmann constant in SI units
MJyperSrtoSI = 1e-20  # MegaJansky/Sr to SI units
GHztoHz = 1e9  # Gigahertz to hertz
HztoGHz = 1e-9  # Hertz to Gigahertz
TCMB = 2.725  # Canonical CMB in Kelvin
m = 9.109 * 10 ** (-31)  # Electron mass in kgs

class Hybrid:
    """
    """
    
    def __init__(self):
        pix1 = 150
        pix2 = 250
        pix3 = 1000
        pix4 = 1250
        pix5 = 750

        res1 = 17
        res2 = 39
        res3 = 26
        res4 = 27
        res5 = 40

        self.bands = [{'name':'Band 1','nu_meanGHz':145,'rms':0.36,'type':'OLIMPO'},\
                      {'name':'Band 2','nu_meanGHz':250,'rms':0.36,'type':'OLIMPO'},\
                      {'name':'Band 3','nu_meanGHz':365,'FBW':0.18,'nu_resGHz':res3,'N_pixels':pix3,'type':'spectrometric'},\
                      {'name':'Band 4','nu_meanGHz':460,'FBW':0.15,'nu_resGHz':res4,'N_pixels':pix4,'type':'spectrometric'}, \
                      {'name':'Band 5','nu_meanGHz':660,'FBW':0.166,'nu_resGHz':res5,'N_pixels':pix5,'type':'spectrometric'}]
        
#     def sigB(band_details, time):
#         """
#         Noise Equivalent Brightness function with known NEPs
#         """

#         BW_GHz = band_details['nu_meanGHz'] * band_details['FBW']

#         nu_min = (band_details['nu_meanGHz'] - 0.5 * BW_GHz) * GHztoHz
#         nu_max = (band_details['nu_meanGHz'] + 0.5 * BW_GHz) * GHztoHz
#         nu_res = band_details['nu_resGHz'] * GHztoHz
#         Npx = band_details['N_pixels']

#         NEP_tot = (band_details['NEP_aWrtHz']) * 1E-18
#         Nse = int(np.round(BW_GHz / band_details['nu_resGHz']))
#         nu_vec = np.linspace(nu_min, nu_max, Nse)
#         AOnu = (c / nu_vec) ** 2

#         # Defined empirically to match OLIMPO inefficiencies at single channel bands
#         inefficiency = 0.019
#         delP = 2.0 * NEP_tot / np.sqrt(time * Npx)
#         sigma_B = delP / AOnu / nu_res / inefficiency

#         return nu_vec, sigma_B

    def sig_b(time, tnoise=3.0):
        """
        Noise Equivalent Brightness function with unknown NEPs.
        Use for apples to apples with OLIMPO photometric mode.
        :param band_details: Bands of instrument
        :param time: Integration time
        :param tnoise: Thermal noise of CMB
        """

        BW_GHz = band_details['nu_meanGHz'] * band_details['FBW']

        nu_min = (band_details['nu_meanGHz'] - 0.5 * BW_GHz) * GHztoHz
        nu_max = (band_details['nu_meanGHz'] + 0.5 * BW_GHz) * GHztoHz
        nu_res = band_details['nu_resGHz'] * GHztoHz
        Npx = band_details['N_pixels']

        NEP_phot1 = NEP.photonNEPdifflim(nu_min=nu_min, nu_max=nu_max, Tsys=tnoise)  # This is CMB Tnoise
        NEP_phot2 = NEP.photonNEPdifflim(nu_min=nu_min, nu_max=nu_max, Tsys=10.0, aef=0.01)  # Use real South Pole data
        NEP_det = 10e-18  # ATTO WATTS per square-root(hz)
        NEP_tot = np.sqrt(NEP_phot1 ** 2 + NEP_phot2 ** 2 + NEP_det ** 2)  # Don't include atmosphere for now

        # in making nu_vec we must be aware of resolution
        Nse = int(np.round(BW_GHz / band_details['nu_resGHz']))
        nu_vec = np.linspace(start=nu_min, stop=nu_max, num=Nse)
        AOnu = (c / nu_vec) ** 2

        # Defined empirically to match OLIMPO inefficiencies at single channel bands
        inefficiency = 0.019
        delP = 2.0 * NEP_tot / np.sqrt(time * Npx)
        sigma_B = delP / AOnu / nu_res / inefficiency

        return nu_vec, sigma_B
    
    def get_sig_b(self, time):
        
        nu_total_array = np.empty(0)
        sigma_b_array = np.empty(0)
        
        for band in self.bands:
            if band['type'] == 'OLIMPO':

            # 80 hour normalized
            rms_value = band['rms'] * (np.sqrt(80) / (np.sqrt(time / 3600)))
            nu_vec_b = band['nu_meanGHz'] * GHztoHz
            x = h_p * nu_vec_b / (k_b * TCMB)
            sigma_B_b = 2 * k_b * ((nu_vec_b / c) ** 2) * (x / (np.exp(x) - 1)) * (x * np.exp(x)) / (
                    np.exp(x) - 1) * rms_value * 1e-6
            nu_total_array = np.concatenate((nu_total_array, nu_vec_b), axis=None)
            sigma_b_array = np.concatenate((sigma_b_array, sigma_B_b), axis=None)

        if band['type'] == 'spectrometric':

            nu_vec_b, sigma_B_b = self.sig_b(band_details=band, time=time)
            nu_total_array = np.concatenate((nu_total_array, nu_vec_b), axis=None)
            sigma_b_array = np.concatenate((sigma_b_array, sigma_B_b), axis=None)
            
        return nu_total_array, sigma_b_array
            
#     # Hybrid 3-3
# pix1 = 150
# pix2 = 250
# pix3 = 1000
# pix4 = 1250
# pix5 = 750

# res1 = 17
# res2 = 39
# res3 = 26
# res4 = 27
# res5 = 40

# spectro_band = [{'name':'Band 1','nu_meanGHz':145,'FBW':0.18,'nu_resGHz':res3,'N_pixels':pix3,'type':'spectrometric'},\
#       {'name':'Band 3','nu_meanGHz':250,'FBW':0.18,'nu_resGHz':res3,'N_pixels':pix3,'type':'spectrometric'},\
#       {'name':'Band 3','nu_meanGHz':365,'FBW':0.18,'nu_resGHz':res3,'N_pixels':pix3,'type':'spectrometric'},\
#       {'name':'Band 4','nu_meanGHz':460,'FBW':0.15,'nu_resGHz':res4,'N_pixels':pix4,'type':'spectrometric'}]

# hybrid_band = [{'name':'Band 1','nu_meanGHz':145,'rms':0.36,'type':'OLIMPO'},\
#       {'name':'Band 2','nu_meanGHz':250,'rms':0.36,'type':'OLIMPO'},\
#       {'name':'Band 3','nu_meanGHz':365,'FBW':0.18,'nu_resGHz':res3,'N_pixels':pix3,'type':'spectrometric'},\
#       {'name':'Band 4','nu_meanGHz':460,'FBW':0.15,'nu_resGHz':res4,'N_pixels':pix4,'type':'spectrometric'}, \
#       {'name':'Band 5','nu_meanGHz':660,'FBW':0.166,'nu_resGHz':res5,'N_pixels':pix5,'type':'spectrometric'}]

# photometric_band = [{'name':'Band 1','nu_meanGHz':145,'rms':0.36,'type':'OLIMPO'},\
#        {'name':'Band 2','nu_meanGHz':250,'rms':0.36,'type':'OLIMPO'},\
#        {'name':'Band 3','nu_meanGHz':365,'rms':0.70,'type':'OLIMPO'},\
#        {'name':'Band 4','nu_meanGHz':460,'rms':1.76,'type':'OLIMPO'}]