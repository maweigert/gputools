from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import numpy.testing as npt
import scipy.ndimage.filters as spf
from itertools import combinations, combinations_with_replacement
from gputools.convolve import  median_filter


def _test_single(dshape, size , cval = 0., dtype = np.float32, skip_assert = False):
    d = np.random.randint(0, 40, dshape).astype(dtype)*0

    out1 = spf.median_filter(d, size, mode = "constant", cval = cval)
    out2 = median_filter(d, size, cval=cval)

    print(("shape: %s \tsize: %s\tcval: %.2f\tdtype: %s\tdifference: %.2f" % (dshape, size,cval, dtype,np.amax(np.abs(out1 - out2)))))
    if not skip_assert:
        npt.assert_almost_equal(out1,out2, decimal = 5)
    return out1, out2


def test_all():
    for ndim in [2,3]:
        for dshape in combinations([19,31,43],ndim):
            for size in combinations_with_replacement([3, 7], ndim):
                for cval in (0,55):
                    for dtype in (np.uint8, np.uint16, np.float32):
                        _test_single(dshape,size, cval = cval, dtype = dtype)

np.random.seed(0)

if __name__ == '__main__':

    x = np.random.randint(0,20,(15,32)).astype(np.float32)*0

    m1= spf.median_filter(x,3,mode = "constant", cval = 10)
    m2 = median_filter(x, 3, cval=10)