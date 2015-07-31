from gputools.core.ocldevice import OCLDevice

class _ocl_globals():
    device = OCLDevice()


def init_device(**kwargs):
    """same arguments as OCLDevice"""
    _ocl_globals.device = OCLDevice(**kwargs)

# _ocl_globals.device.print_info()
    
def get_device():
    return _ocl_globals.device


from gputools.core.ocltypes import OCLArray, OCLImage
from gputools.core.oclprogram import OCLProgram


from gputools.fft.oclfft_convolve import fft_convolve
from gputools.fft.oclfft import fft
from gputools.convolve.convolve_sep import convolve_sep2, convolve_sep3
