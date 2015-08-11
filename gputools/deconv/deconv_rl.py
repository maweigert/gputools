""" Lucy richardson deconvolution
"""
import logging
logger = logging.getLogger(__name__)

import os
import numpy as np
from gputools import OCLArray, OCLProgram, get_device
from gputools import fft_convolve, fft, fft_plan
from gputools import OCLElementwiseKernel

from abspath import abspath



_complex_multiply = OCLElementwiseKernel(
        "cfloat_t *a, cfloat_t * b,cfloat_t * res",
        "res[i] = cfloat_mul(a[i],b[i])",
    "mult")

_complex_multiply_inplace = OCLElementwiseKernel(
        "cfloat_t *a, cfloat_t * b",
        "a[i] = cfloat_mul(a[i],b[i])",
    "mult_inplace")

_complex_divide = OCLElementwiseKernel(
        "cfloat_t *a, cfloat_t * b,cfloat_t * res",
        "res[i] = cfloat_divide(b[i],a[i])",
    "div")

_complex_divide_inplace = OCLElementwiseKernel(
        "cfloat_t *a, cfloat_t * b",
        "b[i] = cfloat_divide(a[i],b[i])",
    "divide_inplace")


def multiply(aBuf,bBuf,resBuf, dev, prog):
    prog.run_kernel("multiply_complex_inplace",(aBuf.size,),None,
                   aBuf.data,bBuf.data,resBuf.data)

def multiply_inplace(aBuf,bBuf,dev, prog):
    prog.run_kernel("multiply_complex_inplace",(aBuf.size,),None,
                   aBuf.data,bBuf.data)

def divide_inplace(aBuf,bBuf,dev, prog):
    prog.run_kernel("divide_complex_inplace",(aBuf.size,),None,
                   aBuf.data,bBuf.data)

def convolve(buf, h_f_buf, resBuf,dev, prog, plan):

    plan.execute(buf.data,resBuf.data)

    prog.run_kernel("multiply_complex_inplace",(buf.size,),None,
                   resBuf.data,h_f_buf.data)

    plan.execute(resBuf.data,inverse=True)

def convolve_inplace(buf, h_f_buf,  dev, prog, plan = None):

    if plan is None:
        plan = Plan(data.shape, queue = dev.queue)

    plan.execute(buf.data)

    prog.run_kernel("multiply_complex_inplace",(buf.size,),None,
                   buf.data,h_f_buf.data)

    plan.execute(buf.data,inverse=True)



def _deconv_rl_np(data, h, Niter = 10, h_is_fftshifted = False):
    """ deconvolves data with given psf (kernel) h

    data and h have to be same shape

    
    via lucy richardson deconvolution
    """


    if data.shape != h.shape:
        raise ValueError("data and h have to be same shape")

    if not h_is_fftshifted:
        h = np.fft.fftshift(h)


    hflip = h[::-1,::-1]
        
    #set up some gpu buffers
    y_g = OCLArray.from_array(data.astype(np.complex64))
    u_g = OCLArray.from_array(data.astype(np.complex64))
    
    tmp_g = OCLArray.empty(data.shape,np.complex64)

    hf_g = OCLArray.from_array(h.astype(np.complex64))
    hflip_f_g = OCLArray.from_array(hflip.astype(np.complex64))

    # hflipped_g = OCLArray.from_array(h.astype(np.complex64))
    
    plan = fft_plan(data.shape)

    #transform psf
    fft(hf_g,inplace = True)
    fft(hflip_f_g,inplace = True)

    for i in range(Niter):
        print i
        fft_convolve(u_g, hf_g,
                     res_g = tmp_g,
                     kernel_is_fft = True)

        _complex_divide_inplace(y_g,tmp_g)

        fft_convolve(tmp_g,hflip_f_g,
                     inplace = True,
                     kernel_is_fft = True)

        _complex_multiply_inplace(u_g,tmp_g)
        

    return np.abs(u_g.get())

def _deconv_rl_gpu(data_g, h_g, Niter = 10):
    """ deconvolves data with given psf (kernel) h

    data and h have to be same shape

    h has to be fft_shifted
    
    via lucy richardson deconvolution
    """


    if data_g.shape != h_g.shape:
        raise ValueError("data and h have to be same shape")

        
    #set up some gpu buffers
    u_g = OCLArray.empty(data_g.shape,np.complex64)

    u_g.copy_buffer(data_g)
    
    tmp_g = OCLArray.empty(data_g.shape,np.complex64)

    #fix this
    hflip_g = OCLArray.from_array((h_g.get()[::-1,::-1]).copy())

    plan = fft_plan(data_g.shape)

    #transform psf
    fft(h_g,inplace = True)
    fft(hflip_g,inplace = True)

    for i in range(Niter):
        print i
        fft_convolve(u_g, h_g,
                     res_g = tmp_g,
                     kernel_is_fft = True)

        _complex_divide_inplace(data_g,tmp_g)

        fft_convolve(tmp_g,hflip_g,
                     inplace = True,
                     kernel_is_fft = True)

        _complex_multiply_inplace(u_g,tmp_g)

    return u_g

if __name__ == '__main__':

    from scipy.misc import lena
    
    d = lena()

    x = np.linspace(-1,1,512+1)[:-1]
    Y,X = np.meshgrid(x,x,indexing="ij")
    
    h = np.exp(-2000*(Y**2+X**2))
    h *= 1./np.sum(h)
    
    y = fft_convolve(d,h)

    y += np.random.normal(0,1,d.shape)

    res = _deconv_rl_np(y,h,20)

    res_g = _deconv_rl_gpu(OCLArray.from_array(y.astype(np.complex64)),
                           OCLArray.from_array(np.fft.fftshift(h).astype(np.complex64)),20)
    
