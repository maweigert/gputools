"""
bilateral filter in 3d

"""

import logging
logger = logging.getLogger(__name__)


import numpy as np

from gputools import OCLArray,OCLImage, OCLProgram, get_device

from _abspath import abspath


def bilateral3(data, size_filter, sigma_p, sigma_x = 10.):
    """bilateral filter """
    
    dtype = data.dtype.type
    dtypes_kernels = {np.float32:"bilat3_float",}

    if not dtype in dtypes_kernels.keys():
        logger.info("data type %s not supported yet (%s), casting to float:"%(dtype,dtypes_kernels.keys()))
        data = data.astype(np.float32)
        dtype = data.dtype.type


    img = OCLImage.from_array(data)
    res = OCLArray.empty_like(data)

    
    prog = OCLProgram(abspath("kernels/bilateral3.cl"))

    print img.shape

    prog.run_kernel(dtypes_kernels[dtype],
                    img.shape,None,
                    img,res.data,
                    np.int32(img.shape[0]),np.int32(img.shape[1]),
                    np.int32(size_filter),np.float32(sigma_x),np.float32(sigma_p))


    return res.get()



if __name__ == '__main__':

    d = 10*np.linspace(0,1,31*32*33).reshape((31,32,33))

    d += np.random.normal(0,1,d.shape)

    res = bilateral3(d,3,100)
