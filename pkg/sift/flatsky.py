import numpy as np


def cl_to_cl2d(el, cl, flatskymapparams):
    """
    converts 1d_cl to 2d_cl
    inputs:
    el = el values over which cl is defined
    cl = power spectra - cl

    flatskymyapparams = [nx, ny, dx, dy] where ny, nx = flatskymap.shape; and dy, dx are the pixel resolution in
     arcminutes.
    for example: [100, 100, 0.5, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with dx = dy = 0.5
    arcminutes.

    output:
    2d_cl
    """
    lx, ly = get_lxly(flatskymapparams)
    ell = np.sqrt(lx ** 2. + ly ** 2.)

    cl2d = np.interp(ell.flatten(), el, cl).reshape(ell.shape)

    return cl2d

def get_lpf_hpf(flatskymapparams, lmin_lmax, filter_type = 0):
    """
    filter_type = 0 - low pass filter
    filter_type = 1 - high pass filter
    filter_type = 2 - band pass
    """

    lx, ly = get_lxly(flatskymapparams)
    ell = np.sqrt(lx**2. + ly**2.)
    fft_filter = np.ones(ell.shape)
    if filter_type == 0:
        fft_filter[ell>lmin_lmax] = 0.
    elif filter_type == 1:
        fft_filter[ell<lmin_lmax] = 0.
    elif filter_type == 2:
        lmin, lmax = lmin_lmax
        fft_filter[ell<lmin] = 0.
        fft_filter[ell>lmax] = 0

    return fft_filter

def get_lxly(flatskymapparams):
    """
    returns lx, ly based on the flatskymap parameters
    input:
    flatskymyapparams = [nx, ny, dx, dy] where ny, nx = flatskymap.shape; and dy, dx are the pixel resolution in
    arcminutes.
    for example: [100, 100, 0.5, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with dx = dy = 0.5
    arcminutes.

    output:
    lx, ly
    """

    nx, ny, dx, dx = flatskymapparams
    dx = np.radians(dx / 60.)

    lx, ly = np.meshgrid(np.fft.fftfreq(nx, dx), np.fft.fftfreq(ny, dx))
    lx *= 2 * np.pi
    ly *= 2 * np.pi

    return lx, ly


def make_gaussian_realisation(mapparams, el, cl, bl=None):
    nx, ny, dx, dy = mapparams
    arcmins2radians = np.radians(1 / 60.)

    dx *= arcmins2radians
    dy *= arcmins2radians

    # map stuff
    norm = np.sqrt(1. / (dx * dy))

    cltwod = cl_to_cl2d(el, cl, mapparams)

    cltwod = cltwod ** 0.5 * norm
    cltwod[np.isnan(cltwod)] = 0.

    gauss_reals = np.random.standard_normal([nx, ny])
    SIM = np.fft.ifft2(np.copy(cltwod) * np.fft.fft2(gauss_reals)).real

    if bl is not None:
        if np.ndim(bl) != 2:
            bl = cl_to_cl2d(el, bl, mapparams)
        SIM = np.fft.ifft2(np.fft.fft2(SIM) * bl).real

    SIM = SIM - np.mean(SIM)

    return SIM
