import logging
logger = logging.getLogger(__name__)

import os
import numpy as np
from gputools import OCLArray, OCLProgram, get_device

from gputools.core.ocltypes import assert_bufs_type

from _abspath import abspath

# def _convolve_axis2_gpu(data_g, h_g, axis= 0, res_g=None, dev = None):
#     if dev is None:
#         dev = get_device()

#     prog = OCLProgram(absPath("kernels/convolve_sep.cl"))

#     N = hy_g.shape[0]

#     tmp_g = OCLArray.empty_like(data_g)

#     kernel_str = "conv_sep2_%s"%(["y","x"][axis],)
    
#     prog.run_kernel(kernel_str,data_g.shape[::-1],None,data_g.data,h_g.data,res_g.data,np.int32(N))

    
def convolve_sep2(data, hx, hy, res_g = None, plan = None):
    """convolves 2d data with kernel h = outer(hx,hy)
    boundary conditions are clamping to edge.

    data is either np array or a gpu buffer (OCLArray)
    
    """
        
    if isinstance(data,np.ndarray):
        return _convolve_sep2_numpy(data, hx, hy)
    elif isinstance(data,OCLArray):
        return _convolve_sep2_gpu(data,hx, hy, res_g = res_g)
    else:
        raise TypeError("array argument (1) has bad type: %s"%type(arr_obj))


def _convolve_sep2_numpy(data,hx,hy):
    hx_g = OCLArray.from_array(hx.astype(np.float32))
    hy_g = OCLArray.from_array(hy.astype(np.float32))

    data_g = OCLArray.from_array(data.astype(np.float32))
                                 

    return _convolve_sep2_gpu(data_g,hx_g,hy_g).get()

def _convolve_sep2_gpu(data_g, hx_g, hy_g, res_g = None, dev = None):

    
    assert_bufs_type(np.float32,data_g,hx_g,hy_g)

    prog = OCLProgram(abspath("kernels/convolve_sep.cl"))

    Ny,Nx = hy_g.shape[0],hx_g.shape[0]

    tmp_g = OCLArray.empty_like(data_g)

    if res_g is None:
        res_g = OCLArray.empty_like(data_g)
    
    prog.run_kernel("conv_sep2_x",data_g.shape[::-1],None,data_g.data,hx_g.data,tmp_g.data,np.int32(Nx))
    prog.run_kernel("conv_sep2_y",data_g.shape[::-1],None,tmp_g.data,hy_g.data,res_g.data,np.int32(Ny))

    return res_g
    

def convolve_sep3(data, hx, hy, hz, res_g = None, plan = None):
    """convolves 3d data with kernel h = outer(hx,hy, hz)
    boundary conditions are clamping to edge.

    data is either np array or a gpu buffer (OCLArray)
    
    """
        
    if isinstance(data,np.ndarray):
        return _convolve_sep3_numpy(data, hx, hy, hz)
    elif isinstance(data,OCLArray):
        return _convolve_sep3_gpu(data,hx, hy, hz, res_g = res_g)
    else:
        raise TypeError("array argument (1) has bad type: %s"%type(data))


def _convolve_sep3_numpy(data,hx,hy,hz):
    
    hx_g = OCLArray.from_array(hx.astype(np.float32))
    hy_g = OCLArray.from_array(hy.astype(np.float32))
    hz_g = OCLArray.from_array(hz.astype(np.float32))

    data_g = OCLArray.from_array(data.astype(np.float32))
                                 
    return _convolve_sep3_gpu(data_g,hx_g,hy_g,hz_g).get()

def _convolve_sep3_gpu(data_g, hx_g, hy_g, hz_g, res_g = None, dev = None):

    assert_bufs_type(np.float32,data_g,hx_g,hy_g)

    prog = OCLProgram(abspath("kernels/convolve_sep.cl"))

    Nz, Ny,Nx = hz_g.shape[0],hy_g.shape[0],hx_g.shape[0]

    tmp_g = OCLArray.empty_like(data_g)

    if res_g is None:
        res_g = OCLArray.empty_like(data_g)

    
    prog.run_kernel("conv_sep3_x",data_g.shape[::-1],None,data_g.data,hx_g.data,res_g.data,np.int32(Nx))
    prog.run_kernel("conv_sep3_y",data_g.shape[::-1],None,res_g.data,hy_g.data,tmp_g.data,np.int32(Ny))
    prog.run_kernel("conv_sep3_z",data_g.shape[::-1],None,tmp_g.data,hy_g.data,res_g.data,np.int32(Nz))

    return res_g


def test_2d():
    import time
    
    data = np.zeros((100,)*2,np.float32)

    data[50,50] = 1.
    hx = 1./5*np.ones(5)
    hy = 1./13*np.ones(13)

    out = convolve_sep2(data,hx,hy)

    data_g = OCLArray.from_array(data)
    hx_g = OCLArray.from_array(hx.astype(np.float32))
    hy_g = OCLArray.from_array(hy.astype(np.float64))

    out_g = convolve_sep2(data_g,hx_g,hy_g)
        
    return  out, out_g.get()
    
def test_3d():
    from time import time
    Niter = 10
    
    data = np.zeros((128,)*3,np.float32)

    data[30,30,30] = 1.
    hx = 1./5*np.ones(5)
    hy = 1./13*np.ones(13)
    hz = 1./13*np.ones(11)

    t = time()
    for _ in range(Niter):
        out = convolve_sep3(data,hx,hy, hz)
    print "time: %.3f ms"%(1000.*(time()-t)/Niter)

    data_g = OCLArray.from_array(data)
    hx_g = OCLArray.from_array(hx.astype(np.float32))
    hy_g = OCLArray.from_array(hy.astype(np.float32))
    hz_g = OCLArray.from_array(hz.astype(np.float32))

    t = time()
    for _ in range(Niter):
        out_g = convolve_sep3(data_g,hx_g,hy_g, hz_g)

    out_g.get();
    print "time: %.3f ms"%(1000.*(time()-t)/Niter)

        
    return  out, out_g.get()
    

if __name__ == '__main__':

    # out1, out2 = test_3d()
    out1, out2 = test_2d()

