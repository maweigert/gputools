from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from gputools import OCLArray
from .convolve import convolve
from .convolve_sep import convolve_sep2, convolve_sep3


def gaussian_filter(data, sigma=4., truncate = 4., strides=1, normalize=True, res_g=None):
    """
    blurs data with a gaussian kernel of given sigmas
    
    Parameters
    ----------
    data: ndarray
        2 or 3 dimensional array 
    sigma: scalar or tuple
        the sigma of the gaussian  
    truncate: float 
        truncate the kernel after truncate*sigma  
    normalize: bool
        uses a normalized kernel is true
    res_g: OCLArray
        used to store result if given  

    Returns
    -------
        blurred array 
    """

    if not len(data.shape) in [1, 2, 3]:
        raise ValueError("dim = %s not supported" % (len(data.shape)))

    if np.isscalar(sigma):
        sigma = (sigma,) * data.ndim

    if np.isscalar(strides):
        strides = (strides,)*data.ndim        
    if any(tuple(s < 0 for s in sigma)):
        raise ValueError("sigma = %s : all sigmas have to be non-negative!" % str(sigma))

    if isinstance(data, OCLArray):
        return _gaussian_buf(data, sigma, res_g, normalize=normalize,strides=strides, truncate = truncate)
    elif isinstance(data, np.ndarray):
        return _gaussian_np(data, sigma,  normalize=normalize,strides=strides, truncate = truncate)

    else:
        raise TypeError("unknown type (%s)" % (type(data)))


def _gaussian_buf(d_g, sigma=(4., 4.),  res_g=None, normalize=True,strides=1, truncate = 4.0):
    # https://github.com/scipy/scipy/blob/053f3f54130f8b17c09540352a784f4c54a919df/scipy/ndimage/_filters.py#L211
    
    radius = tuple(int(truncate*s +0.5) for s in sigma)
    ns = tuple(np.arange(-r,r+1) for r in radius)

    hs = tuple(
        np.exp(-.5 / max(s,1e-10) ** 2 * n**2) for s, n in zip(reversed(sigma), reversed(ns)))

    if normalize:
        hs = tuple(1. * h / np.sum(h) for h in hs)

    h_gs = tuple(OCLArray.from_array(h.astype(np.float32)) for h in hs)

    if len(d_g.shape) == 1:
        if max(strides) != 1:
            raise NotImplementedError('strides not implemented 1D arrays')
        return convolve(d_g, *h_gs, res_g=res_g)
    elif len(d_g.shape) == 2:
        return convolve_sep2(d_g, *h_gs, res_g=res_g, strides=strides)
    elif len(d_g.shape) == 3:
        return convolve_sep3(d_g, *h_gs, res_g=res_g, strides=strides)
    else:
        raise NotImplementedError("only 1D, 2D, or 3D images supported yet")


def _gaussian_np(data, sigma,  normalize=True, truncate = 4.0, strides=1):
    d_g = OCLArray.from_array(data)

    return _gaussian_buf(d_g, sigma, truncate = truncate, normalize=normalize, strides=strides).get()


if __name__ == "__main__":
    pass
