from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import numpy.testing as npt
from itertools import product, combinations_with_replacement
import scipy.ndimage.filters as spf
import gputools

from gputools.convolve import  max_filter, min_filter, uniform_filter


np.random.seed(0)


def _test_single(filt1, filt2, dshape, size , cval = 0., dtype = np.float32, strides=(1,1), skip_assert = False):
    d = np.random.uniform(-1, 1., dshape).astype(dtype)

    ss_stride = tuple(slice(0,None,s) for s in strides)
    out1 = filt1(d, size, strides=strides)
    out2 = filt2(d, size, mode = "constant", cval = cval)[ss_stride]
    print(("shape: %s \tsize: %s\t stride: %s \tdifference: %s" % (dshape, size, strides, np.amax(np.abs(out1 - out2)))))
    if not skip_assert:
        npt.assert_almost_equal(out1,out2, decimal = 3)
    else:
        print(np.allclose(out1,out2, rtol=1e-2))
    return d, out1, out2


def _test_some(filt1, filt2, cval = 0.):
    stridess = {2:((1,1),(2,2),(4,3)), 3:((1,1,1),(2,2,2),(4,1,1),(3,2,5))}
    for ndim in [2,3]:
        for dshape in combinations_with_replacement([32,44,53],ndim):
            for size in [3,7,13]:
                for dtype in (np.uint8, np.uint16, np.float32):
                    for strides in stridess[ndim]:
                        _test_single(filt1, filt2, dshape,size, cval = cval, strides=strides, dtype = dtype)

def test_all():
    print("~"*40, " maximum filter")
    _test_some(max_filter, spf.maximum_filter, cval = -np.inf)
    print("~" * 40, " minimum filter")
    _test_some(min_filter, spf.minimum_filter, cval = np.inf)
    print("~" * 40, " uniform filter")
    _test_some(uniform_filter, spf.uniform_filter, cval = 0.)


if __name__ == '__main__':
    # _test_some(uniform_filter, spf.uniform_filter, cval = 0.)
    # _test_some(max_filter, spf.maximum_filter, cval = -np.inf)
    # _test_some(min_filter, spf.minimum_filter, cval=np.inf)

    test_all()

    # np.random.seed(27)
    # x, a,b = _test_single(uniform_filter, spf.uniform_filter, (32,53, 15), 3, strides=(2,)*3, cval=0, skip_assert=True)
