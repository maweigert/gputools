import numpy as np

from pyfft.cl import Plan

from gputools import OCLArray, get_device


def fft_plan(shape):
    return Plan(shape, queue = get_device().queue)


# def fft(arr, inplace = False, inverse = False, plan = None):
#     if isinstance(arr,np.ndarray):
#         return _ocl_fft_np(arr,inverse = inverse, plan = plan)
#     else:
#         raise TypeError("array argument (1) has bad type: %s"%type(arr))

# def fft_g(arr_g,res_g = None, inplace = False, inverse = False, plan = None):
#     if isinstance(arr_g,OCLArray):
#         if inplace:
#             _ocl_fft_buffer_inplace(arr_g,inverse = inverse, plan = plan)
#         else:
#             return _ocl_fft_buffer(arr_g,res_g,inverse = inverse, plan = plan)

#     else:
#         raise TypeError("array argument (1) has bad type: %s"%type(arr_g))

    

def fft(arr_obj,res_g = None, inplace = False, inverse = False, plan = None):
    """ fourier trafo of 1-3D arrays

    arr_obj should be either a numpy array or a complex64 OCLArray 

    returns either a numpy array or the  GPU array res_arr when given 
    """
    
    if isinstance(arr_obj,np.ndarray):
        return _ocl_fft_numpy(arr_obj,inverse = inverse, plan = plan)
    elif isinstance(arr_obj,OCLArray):
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

    ocl_arr = OCLArray.from_array(arr)
    plan.execute(ocl_arr.data,ocl_arr.data, inverse = inverse)
    return ocl_arr.get()

def _ocl_fft_gpu_inplace(ocl_arr,inverse = False, plan = None):

    if ocl_arr.dtype.type != np.complex64: 
        raise TypeError("arraygpu buffer argument has bad type: %s but should be complex64"%ocl_arr.dtype)

    if plan is None:
        plan = Plan(ocl_arr.shape, queue = get_device().queue)

    plan.execute(ocl_arr.data,ocl_arr.data, inverse = inverse)

def _ocl_fft_gpu(ocl_arr,res_arr,inverse = False, plan = None):

    if ocl_arr.dtype.type != np.complex64: 
        raise TypeError("arraygpu buffer argument has bad type: %s but should be complex64"%ocl_arr.dtype)

    if plan is None:
        plan = Plan(ocl_arr.shape, queue = get_device().queue)

    plan.execute(ocl_arr.data,res_arr.data, inverse = inverse)

    return res_arr



if __name__ == '__main__':


    d = np.random.uniform(0,1,(64,)*3).astype(np.complex64)

    b = OCLArray.from_array(d)

    plan  = fft_plan(d.shape)
    
    d2 = fft(d, plan = plan)
    
    fft(b, inplace = True, plan = plan)
             
    
