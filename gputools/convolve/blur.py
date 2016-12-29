from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from gputools import OCLArray
from .convolve import convolve
from .convolve_sep import convolve_sep2, convolve_sep3


def blur(data, width = 4., res_g  = None):
    """ blurs data with a gaussian kernel of given width (i.e. sigma = width/2)

    width is either a scalar or a list of widths (of length data.ndim)
    """

    
    if not len(data.shape) in [1,2,3]:
        raise ValueError("dim = %s not supported"%(len(data.shape)))

    if np.isscalar(width):
        width = [width]*data.ndim
        
    if isinstance(data,OCLArray):
        return _blur_buf(data, width,res_g)
    elif isinstance(data,np.ndarray):
        return _blur_np(data,width)
    
    else:
        raise TypeError("unknown type (%s)"%(type(data)))

    return 


def _blur_buf(d_g,width = (4.,4.), res_g = None ):

    Ns = [3*s+1 for s in width]
    sigmas = [.5*s for s in width]
    
    hs = [np.exp(-.5/s**2*np.linspace(-N/2,N/2,N)**2) for s,N in zip(sigmas,Ns)]
    
    h_gs = [OCLArray.from_array(h.astype(np.float32)) for h in hs][::-1]

    if len(d_g.shape) == 1:
        return convolve(d_g,*h_gs, res_g = res_g)
    elif len(d_g.shape) == 2:
        return convolve_sep2(d_g,*h_gs, res_g = res_g)
    elif len(d_g.shape) == 3:
        return convolve_sep3(d_g,*h_gs, res_g = res_g)

    else:
        pass

    
def _blur_np(data,width):

    d_g = OCLArray.from_array(data.astype(np.float32,copy=False))
    
    return _blur_buf(d_g,width).get()



if __name__ == "__main__":
    pass
