
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import numpy.testing as npt
from itertools import product
from termcolor import colored

from gputools import fft, fft_convolve, fft_plan, init_device

#init_device(id_platform = 0, id_device = 1)

def report_str(success):
    return colored("\t[OK]", "blue") if success else colored("\t[FAIL]", "red")

def _compare_fft_np(d):
    res1 = np.fft.fftn(d)
    res2 = fft(d, fast_math=True)
    return res1, res2

def test_compare():
    for ndim in [1, 2, 3]:
        for dshape in product([32, 64, 128], repeat=ndim):
            d = np.random.uniform(-1, 1, dshape).astype(np.complex64)
            res1, res2 = _compare_fft_np(d)
            print("validating fft of size", d.shape)
            npt.assert_allclose(res1, res2, rtol=1.e-0, atol=1.e-1)





if __name__ == '__main__':
    # test_compare()
    #
    dshape = (128, 128)
    np.random.seed(0)
    d = np.random.uniform(-1, 1, dshape).astype(np.complex64)
    res1 = np.fft.fftn(d)
    res2 = fft(d)
