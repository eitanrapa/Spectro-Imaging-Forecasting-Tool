import numpy as np
from .Mather_photonNEP12a import photonNEPdifflim

# Constants
c = 299792458.0  # Speed of light - [c] = m/s
h_p = 6.626068e-34  # Planck's constant in SI units
k_b = 1.38065e-23  # Boltzmann constant in SI units
MJyperSrtoSI = 1e-20  # MegaJansky/Sr to SI units
GHztoHz = 1e9  # Gigahertz to hertz
HztoGHz = 1e-9  # Hertz to Gigahertz
TCMB = 2.725  # Canonical CMB in Kelvin
m = 9.109 * 10 ** (-31)  # Electron mass in kgs


class Band:
    """
    A band class that encapsulates the instrument NESBs
    """

    def __init__(self, bands):
        self.bands = bands

    #     def sigB(self.bands, time):
    #         """
    #         Noise Equivalent Brightness function with known NEPs
    #         """

    #         BW_GHz = self.bands['nu_meanGHz'] * self.bands['FBW']

    #         nu_min = (self.bands['nu_meanGHz'] - 0.5 * BW_GHz) * GHztoHz
    #         nu_max = (self.bands['nu_meanGHz'] + 0.5 * BW_GHz) * GHztoHz
    #         nu_res = self.bands['nu_resGHz'] * GHztoHz
    #         Npx = self.bands['N_pixels']

    #         NEP_tot = (self.bands['NEP_aWrtHz']) * 1E-18
    #         Nse = int(np.round(BW_GHz / self.bands['nu_resGHz']))
    #         nu_vec = np.linspace(nu_min, nu_max, Nse)
    #         AOnu = (c / nu_vec) ** 2

    #         # Defined empirically to match OLIMPO inefficiencies at single channel bands
    #         inefficiency = 0.019
    #         delP = 2.0 * NEP_tot / np.sqrt(time * Npx)
    #         sigma_B = delP / AOnu / nu_res / inefficiency

    #         return nu_vec, sigma_B

    def sig_b(self, band, time, tnoise=3.0):
        """
        Noise Equivalent Brightness function with unknown NEPs.
        Use for apples to apples with OLIMPO photometric mode
        :param band:
        :param time: Integration time
        :param tnoise: Thermal noise of CMB
        """

        BW_GHz = band['nu_meanGHz'] * band['FBW']

        nu_min = (band['nu_meanGHz'] - 0.5 * BW_GHz) * GHztoHz
        nu_max = (band['nu_meanGHz'] + 0.5 * BW_GHz) * GHztoHz
        nu_res = band['nu_resGHz'] * GHztoHz
        Npx = band['N_pixels']

        NEP_phot1 = photonNEPdifflim(nu_min=nu_min, nu_max=nu_max, Tsys=tnoise)  # This is CMB Tnoise
        NEP_phot2 = photonNEPdifflim(nu_min=nu_min, nu_max=nu_max, Tsys=10.0, aef=0.01)  # Use real South Pole data
        NEP_det = 10e-18  # ATTO WATTS per square-root(hz)
        NEP_tot = np.sqrt(NEP_phot1 ** 2 + NEP_phot2 ** 2 + NEP_det ** 2)  # Don't include atmosphere for now

        # in making nu_vec we must be aware of resolution
        Nse = int(np.round(BW_GHz / band['nu_resGHz']))
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
                nu_vec_b, sigma_B_b = self.sig_b(time=time, band=band)
                nu_total_array = np.concatenate((nu_total_array, nu_vec_b), axis=None)
                sigma_b_array = np.concatenate((sigma_b_array, sigma_B_b), axis=None)

        return nu_total_array, sigma_b_array
