import logging
logger = logging.getLogger(__name__)

import os
import numpy as np

from gputools import  OCLProgram, OCLArray, OCLImage, get_device

import sys

from abspath import abspath




def convolve(data,h):
    """convolves 1-3d data with kernel h on the GPU Device dev
    boundary conditions are clamping to edge.
    h is converted to float32

    if dev == None the default one is used
    """

    if not data.ndim in [1,2,3]:
        raise ValueError("dim = %s not supported"%(data.ndim))

    if not data.dtype.type in [np.float32,np.uint16]:
        print "dtype %s not supported, casting to float32..."%(data.dtype.type)
        _data = data.astype(np.float32)
    else:
        _data = data

    if _data.ndim == 1:
        return _convolve1(_data,h)
    if _data.ndim == 2:
        return _convolve2(_data,h)
    if _data.ndim == 3:
        return _convolve3(_data,h)
            
        
    
def _convolve1(data,h, dev = None):
    """convolves 1d data with kernel h on the GPU Device dev
    boundary conditions are clamping to edge.
    h is converted to float32

    if dev == None the default one is used
    """


    dtype = data.dtype.type

    dtypes_options = {np.float32:"",
                      np.uint16:"-D SHORTTYPE"}

    if not dtype in dtypes_options.keys():
        raise TypeError("data type %s not supported yet, please convert to:"%dtype,dtypes_options.keys())

    prog = OCLProgram(abspath("kernels/convolve1.cl"),
                      build_options = dtypes_options[dtype])

    
    hbuf = OCLArray.from_array(h.astype(np.float32))
    img = OCLImage.from_array(data.reshape((1,)+data.shape))
    res = OCLArray.empty(data.shape,dtype=np.float32)

    Ns = [np.int32(n) for n in data.shape+h.shape]

    prog.run_kernel("convolve1d",img.shape,None,
                    img,hbuf.data,res.data,
                    *Ns)

    return res.get()

def _convolve2(data,h):
    """convolves 2d data with kernel h on the GPU Device dev
    boundary conditions are clamping to edge.
    h is converted to float32
    """

    dtype = data.dtype.type

    dtypes_options = {np.float32:"",
                      np.uint16:"-D SHORTTYPE"}

    if not dtype in dtypes_options.keys():
        raise TypeError("data type %s not supported yet, please convert to:"%dtype,dtypes_options.keys())

    prog = OCLProgram(abspath("kernels/convolve2.cl"),
                      build_options = dtypes_options[dtype])

    hbuf = OCLArray.from_array(h.astype(np.float32))
    img = OCLImage.from_array(data)
    res = OCLArray.empty(data.shape,dtype=np.float32)

    Ns = [np.int32(n) for n in data.shape+h.shape]
    
    prog.run_kernel("convolve2d",img.shape,None,
                    img,hbuf.data,res.data,
                    *Ns)

    return res.get()


def _convolve3(data,h, dev = None):
    """convolves 3d data with kernel h on the GPU Device dev
    boundary conditions are clamping to edge.
    h is converted to float32

    if dev == None the default one is used
    """

    if dev is None:
        dev = get_device()

    if dev is None:
        raise ValueError("no OpenCLDevice found...")

    dtype = data.dtype.type

    dtypes_options = {np.float32:"",
                      np.uint16:"-D SHORTTYPE"}

    if not dtype in dtypes_options.keys():
        raise TypeError("data type %s not supported yet, please convert to:"%dtype,dtypes_options.keys())

    prog = OCLProgram(abspath("kernels/convolve3.cl"),
                      build_options = dtypes_options[dtype])

    
    hbuf = OCLArray.from_array(h.astype(np.float32))
    img = OCLImage.from_array(data)
    res = OCLArray.empty(data.shape,dtype=np.float32)

    Ns = [np.int32(n) for n in data.shape+h.shape]

    prog.run_kernel("convolve3d",img.shape,None,
                    img,hbuf.data,res.data,
                    *Ns)

    return res.get()




def test_convolve():
    from time import time
    
    data  = np.ones((100,120,140))
    h = np.ones((10,11,12))

    # out = convolve(data,h)
    out = convolve(data[0,...],h[0,...])
    out = convolve(data[0,0,...],h[0,0,...])
    


if __name__ == '__main__':

    test_convolve()
    
