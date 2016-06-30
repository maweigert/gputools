""" some image manipulation functions like scaling, rotating, etc...

"""


import logging
logger = logging.getLogger(__name__)

import os
import numpy as np
from gputools import OCLArray, OCLImage, OCLProgram, get_device
from gputools import OCLElementwiseKernel

from _abspath import abspath


def scale(data, scale = (1.,1.,1.), interp = "linear"):
    """returns a interpolated, scaled version of data

    scale = (scale_z,scale_y,scale_x)
    or
    scale = scale_all

    interp = "linear" | "nearest"
    """ 

    bop = {"linear":[],"nearest":["-D","USENEAREST"]}

    if not interp in bop.keys():
        raise KeyError("interp = '%s' not defined ,valid: %s"%(interp,bop.keys()))
    
    if not isinstance(scale,(tuple, list, np.ndarray)):
        scale = (scale,)*3

    if len(scale) != 3:
        raise ValueError("scale = %s misformed"%scale)

    d_im = OCLImage.from_array(data)

    nshape = np.array(data.shape)*np.array(scale)
    nshape = tuple(nshape.astype(np.int))

    res_g = OCLArray.empty(nshape,np.float32)


    prog = OCLProgram(abspath("kernels/scale.cl"), build_options=bop[interp])


    prog.run_kernel("scale",
                    res_g.shape[::-1],None,
                    d_im,res_g.data)

    return res_g.get()




if __name__ == '__main__':
    
    d = np.zeros((100,100,100),np.float32)
    d[10:-10,10:-10,10:-10] = 1.

    res = scale(d,1.7, interp = "linear")
    # res = scale(d)

    
    
