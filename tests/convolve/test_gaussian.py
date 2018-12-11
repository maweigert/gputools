from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import numpy.testing as npt
import scipy.ndimage.filters as spf
from itertools import combinations_with_replacement, combinations
from gputools.convolve import gaussian_filter


def _test_single(dshape, sigma, dtype = np.float32, skip_assert = False):
    x = np.random.randint(0, 40, dshape).astype(dtype)

    out1 = spf.gaussian_filter(x, sigma, mode= "constant", cval=0)
    out2 = gaussian_filter(x, sigma)

    print(("shape: %s \tsigma: %s\ttype: %s\tdifference: %.2f" % (dshape, sigma, dtype,np.amax(np.abs(out1 - out2)))))
    if not skip_assert:
        npt.assert_almost_equal(out1,out2, decimal = 3)
    return out1, out2


def test_all():
    for ndim in [2,3]:
        for dshape in combinations([19,31,43],ndim):
            for sigma in combinations_with_replacement([3,4,5],ndim):
                for dtype in (np.float32,):
                        _test_single(dshape,sigma, dtype = dtype)

    
if __name__ == '__main__':
    x,y = _test_single((200,100),(3,4), dtype = np.float32)

   

