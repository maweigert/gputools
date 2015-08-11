import logging
logging.basicConfig(level=logging.INFO)

from gputools.core.ocldevice import OCLDevice


class _ocl_globals():
    device = OCLDevice()

def init_device(**kwargs):
    """same arguments as OCLDevice"""
    _ocl_globals.device = OCLDevice(**kwargs)
    
def get_device():
    return _ocl_globals.device


