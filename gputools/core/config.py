import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import pyopencl

from gputools.core.ocldevice import OCLDevice

cl_datatype_dict = {pyopencl.channel_type.FLOAT:np.float32,
                    pyopencl.channel_type.UNSIGNED_INT8:np.uint8,
                    pyopencl.channel_type.UNSIGNED_INT16:np.uint16,
                    pyopencl.channel_type.SIGNED_INT8:np.int8,
                    pyopencl.channel_type.SIGNED_INT16:np.int16,
                    pyopencl.channel_type.SIGNED_INT32:np.int32}

cl_datatype_dict.update({dtype:cltype for cltype,dtype in cl_datatype_dict.iteritems()})



class _ocl_globals():
    device = OCLDevice()

def init_device(**kwargs):
    """same arguments as OCLDevice
    e.g. useDevice = 1
    """
    new_device = OCLDevice(**kwargs)

    #just change globals if new_device is different from old
    if _ocl_globals.device.device != new_device.device:
        _ocl_globals.device = new_device
    
def get_device():
    return _ocl_globals.device


