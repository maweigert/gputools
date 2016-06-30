import logging
logging.basicConfig(format='%(levelname)s:%(name)s | %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


from gputools.core.config import init_device, get_device

from gputools.core.ocltypes import OCLArray, OCLImage
from gputools.core.oclprogram import OCLProgram

from gputools.fft.oclfft_convolve import fft_convolve
from gputools.fft.oclfft import fft, fft_plan

from gputools.convolve.convolve_sep import convolve_sep2, convolve_sep3
from gputools.convolve.convolve import convolve
from gputools.convolve.blur import blur


from gputools.core.oclalgos import OCLReductionKernel, OCLElementwiseKernel

from gputools.noise import perlin2, perlin3

from gputools import denoise
from gputools import deconv
from gputools import convolve
from gputools import transforms

from gputools import noise

from gputools.transforms import scale
from gputools.transforms import affine, rotate, translate

from gputools.utils.utils import pad_to_shape, pad_to_power2
from gputools.utils.utils import remove_cache_dir
