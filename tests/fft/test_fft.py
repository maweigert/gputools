from __future__ import print_function, unicode_literals, absolute_import, division

import numpy as np
import numpy.testing as npt
from itertools import product

from gputools import fft, fft_convolve, fft_plan



def _compare_fft_np(dshape):
    d = np.random.uniform(-1,1,dshape).astype(np.complex64)
    res1 = np.fft.fftn(d)
    res2 = fft(d)
    print("validating fft of size", dshape)
    npt.assert_allclose(res1,res2, rtol = 1.e-2)

def test_compare():
    for ndim in [1,2,3]:
        for dshape in product([32, 64,128], repeat = ndim):
            _compare_fft_np(dshape)


if __name__ == '__main__':

    # test_compare()
    #
    dshape = (128,128)
    np.random.seed(0)
    d = np.random.uniform(-1,1,dshape).astype(np.complex64)
    res1 = np.fft.fftn(d)
    res2 = fft(d)

    



