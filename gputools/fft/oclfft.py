import numpy as np

from pyfft.cl import Plan

import logging 
logger = logging.getLogger(__name__)

from gputools import OCLArray, get_device
from gputools.core.ocltypes import assert_bufs_type



def fft_plan(shape, **kwargs):
    """returns an opencl/pyfft plan of shape dshape

    kwargs are the same as pyfft.cl.Plan
    """
    return Plan(shape, queue = get_device().queue, **kwargs)


    

def fft(arr_obj,res_g = None, inplace = False, inverse = False, plan = None):
    """ (inverse) fourier trafo of 1-3D arrays

    creates a new plan or uses the given plan
    
    the transformed arr_obj should be either a

    - numpy array:

        returns the fft as numpy array (inplace is ignored)
    
    - OCLArray of type complex64:

        writes transform into res_g if given, to arr_obj if inplace
        or returns a new OCLArray with the transform otherwise
    
    """
    
    if isinstance(arr_obj,np.ndarray):
        return _ocl_fft_numpy(arr_obj,inverse = inverse, plan = plan)
    elif isinstance(arr_obj,OCLArray):
        if not arr_obj.dtype.type is np.complex64:
            raise TypeError("OCLArray arr_obj has to be of complex64 type")
        
        if inplace:
            _ocl_fft_gpu_inplace(arr_obj,inverse = inverse, plan = plan)
        else:
            return _ocl_fft_gpu(arr_obj,res_g,inverse = inverse, plan = plan)

    else:
        raise TypeError("array argument (1) has bad type: %s"%type(arr_obj))


# implementation ------------------------------------------------

def _ocl_fft_numpy(arr,inverse = False, plan = None):
    if plan is None:
        plan = Plan(arr.shape, queue = get_device().queue)

    if arr.dtype != np.complex64:
       logger.info("converting %s to complex64, might slow things down..."%arr.dtype)

    ocl_arr = OCLArray.from_array(arr.astype(np.complex64,copy=False))
    
    _ocl_fft_gpu_inplace(ocl_arr, inverse = inverse, plan  = plan)
    
    return ocl_arr.get()
    
def _ocl_fft_gpu_inplace(ocl_arr,inverse = False, plan = None):

    assert_bufs_type(np.complex64,ocl_arr)

    if plan is None:
        plan = Plan(ocl_arr.shape, queue = get_device().queue)

    plan.execute(ocl_arr.data,ocl_arr.data, inverse = inverse)

def _ocl_fft_gpu(ocl_arr,res_arr = None,inverse = False, plan = None):

    assert_bufs_type(np.complex64,ocl_arr)

    if plan is None:
        plan = Plan(ocl_arr.shape, queue = get_device().queue)

    if res_arr is None:
        res_arr = OCLArray.empty(ocl_arr.shape,np.complex64)
        
    plan.execute(ocl_arr.data,res_arr.data, inverse = inverse)

    return res_arr



if __name__ == '__main__':


    d = np.random.uniform(0,1,(64,)*3).astype(np.complex64)

    b = OCLArray.from_array(d)

    plan  = fft_plan(d.shape)
    
    d2 = fft(d, plan = plan)
    
    fft(b, inplace = True, plan = plan)
             
    
