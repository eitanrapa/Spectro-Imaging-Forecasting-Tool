# -*- Python -*-
# -*- coding: utf-8 -*-
#
# the sift development team
# california institute of technology
# (c) 2023-2025 all rights reserved
#

from .flatsky import make_gaussian_realisation
import scipy as sc
from pylab import *


def calccov(sim_mat, noofsims, npixels):
    m = sim_mat.flatten().reshape(noofsims, npixels)
    m = np.asmatrix(m).T
    mt = m.T

    cov = (m * mt) / noofsims
    return cov


#################################################################################
def get_mask_indices(ra_grid, dec_grid, mask_radius_inner, mask_radius_outer, square=0, in_arcmins=1):
    if not in_arcmins:
        ra_grid = ra_grid * 60.
        dec_grid = dec_grid * 60.

    if not square:
        radius = np.sqrt((ra_grid ** 2. + dec_grid ** 2.))
        inds_inner = np.where((radius <= mask_radius_inner))
        inds_outer = np.where((radius > mask_radius_inner) & (radius <= mask_radius_outer))
    else:
        inds_inner = np.where((abs(ra_grid) <= mask_radius_inner) & (abs(dec_grid) <= mask_radius_inner))
        inds_outer = np.where((abs(ra_grid) <= mask_radius_outer) & (abs(dec_grid) <= mask_radius_outer) & (
                (abs(ra_grid) > mask_radius_inner) | (abs(dec_grid) > mask_radius_inner)))

    return inds_inner, inds_outer


def get_covariance(ra_grid, dec_grid, mapparams, el, cl_dic, bl, nl_dic, noofsims, mask_radius_inner, mask_radius_outer, lpf,
                   low_pass_cutoff=1):
    print('\n\tcalculating the covariance from simulations for inpainting')

    # get the sims for covariance calculation
    print('\n\t\tgenerating %s sims' % noofsims)

    sims_for_covariance = []
    for n in range(noofsims):

        # cmb sim and beam, for CMB include the transfer function and beam
        cmb_map = make_gaussian_realisation(mapparams, el, cl_dic['TT'], bl=bl)

        noise_map = make_gaussian_realisation(mapparams, el, nl_dic['T'])
        sim_map = cmb_map + noise_map

        # lpf the map
        if low_pass_cutoff:
            sim_map = np.fft.ifft2(np.fft.fft2(sim_map) * lpf).real

        sims_for_covariance.append(sim_map)

    sims_for_covariance = np.asarray(sims_for_covariance)

    # get the inner and outer pixel indices
    inds_inner, inds_outer = get_mask_indices(ra_grid, dec_grid, mask_radius_inner, mask_radius_outer)

    # get the pixel values in the inner and outer regions
    t1_for_cov = sims_for_covariance[:, inds_inner[0], inds_inner[1]]
    t2_for_cov = sims_for_covariance[:, inds_outer[0], inds_outer[1]]

    # get the covariance now
    npixels_t1 = t1_for_cov.shape[1]

    t1t2_for_cov = np.concatenate((t1_for_cov, t2_for_cov), axis=1)
    npixels_t1t2 = t1t2_for_cov.shape[1]
    t1t2_cov = calccov(t1t2_for_cov, noofsims, npixels_t1t2)

    sigma_22 = t1t2_cov[npixels_t1:, npixels_t1:]
    sigma_12 = t1t2_cov[:npixels_t1, npixels_t1:]

    print('\n\t\t\tinvert sigma_22 matrix (%s,%s) now' % (sigma_22.shape[0], sigma_22.shape[1]))
    sigma_22_inv = sc.linalg.pinv(sigma_22)
    sigma_dic = {'sigma_22_inv': sigma_22_inv, 'sigma_12': sigma_12}

    print('\n\t\tcovariance obtained')
    return sigma_dic


def inpainting(map_dic_to_inpaint, ra_grid, dec_grid, mapparams, el, cl_dic, bl, nl_dic, mask_radius_inner, lpf,
               mask_radius_outer, low_pass_cutoff=1, sigma_dic=None, use_original=False):
    """
    mask_inner = 1: The inner region is masked before the LPF. Might be useful in the presence of bright SZ signal at
     the centre.
    """

    sigma_12 = sigma_dic['sigma_12']
    sigma_22_inv = sigma_dic['sigma_22_inv']

    # get the inner and outer pixel indices
    inds_inner, inds_outer = get_mask_indices(ra_grid, dec_grid, mask_radius_inner, mask_radius_outer)

    tqukeys = ['T']

    map_to_inpaint = []
    for tqu in tqukeys:
        map_to_inpaint.append(map_dic_to_inpaint[tqu])
    map_to_inpaint = np.asarray(map_to_inpaint)

    original_map = map_to_inpaint.copy()

    # lpf the map
    if low_pass_cutoff:
        map_to_inpaint = np.fft.ifft2(np.fft.fft2(map_to_inpaint) * lpf).real

    # get the pixel values in the inner and outer regions
    map_to_inpaint = map_to_inpaint.reshape(np.shape(map_to_inpaint)[2], np.shape(map_to_inpaint)[2])
    t2_data = map_to_inpaint[inds_outer[0], inds_outer[1]].flatten()

    # generate constrained Gaussian CMB realisation now
    # include transfer function for the data map
    # include noise twod for the noise

    '''
    cmb_map = make_gaussian_realisation(mapparams, el, cl, bl = bl) #cmb sim and beam
    noise_map = make_gaussian_realisation(mapparams, el, nl) #noise map
    '''

    cmb_map = np.asarray([make_gaussian_realisation(mapparams, el, cl_dic['TT'], bl=bl)])
    noise_map = np.asarray([make_gaussian_realisation(mapparams, el, nl_dic['T'])])

    constrained_sim_to_inpaint = cmb_map + noise_map  # combined
    # lpf the map
    if low_pass_cutoff:
        constrained_sim_to_inpaint = np.fft.ifft2(np.fft.fft2(constrained_sim_to_inpaint) * lpf).real

    t1_tilde = constrained_sim_to_inpaint[:, inds_inner[0], inds_inner[1]].flatten()
    t2_tilde = constrained_sim_to_inpaint[:, inds_outer[0], inds_outer[1]].flatten()

    # get the modified t1 values
    inpainted_t1 = np.asarray(t1_tilde + np.dot(sigma_12, np.dot(sigma_22_inv, (t2_data - t2_tilde))))[0]  # Eq. 36

    # create a new inpainted map: copy the old map and replace the t1 region
    # if use_original is True, put the inpainted region in the input map before lpf
    if use_original:
        map_to_inpaint = original_map
    inpainted_map = np.copy(map_to_inpaint)

    # split inpainted T(or /Q/U) vector in nx1(or 3) array
    npixels_t1 = int(len(t1_tilde) / len(tqukeys))
    inpainted_t1_tqu_split = inpainted_t1.reshape(len(tqukeys), npixels_t1)
    inpainted_map[inds_inner[0], inds_inner[1]] = inpainted_t1_tqu_split

    cmb_inpainted_map = np.copy(map_to_inpaint) * 0.
    cmb_inpainted_map[inds_inner[0], inds_inner[1]] = inpainted_t1_tqu_split

    return cmb_inpainted_map, inpainted_map, map_to_inpaint

# end of file
