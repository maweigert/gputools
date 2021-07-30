from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import numpy.testing as npt
import scipy.ndimage.filters as spf
from itertools import combinations_with_replacement, combinations
from gputools.convolve import gaussian_filter
from gputools.convolve.generic_separable_filters import _gauss_filter


def _test_single(dshape, sigma, dtype = np.float32, strides=(1,1),skip_assert = False):
    x = np.random.randint(0, 240, dshape).astype(dtype)

    ss_stride = tuple(slice(0,None,s) for s in strides)
    
    out1 = gaussian_filter(x, sigma, strides=strides)
    out2 = spf.gaussian_filter(x, sigma, mode= "constant", cval=0)[ss_stride]

    print(("shape: %s sigma: %s  strides %s  type: %s  diff: %.2f" % (dshape, sigma, strides, dtype,np.amax(np.abs(1.*out1 - out2)))))
    if not skip_assert:
        npt.assert_almost_equal(out1,out2, decimal = 0)
    return out1, out2


def test_all():
    stridess = {2:((1,1),(2,2),(4,3)), 3:((1,1,1),(2,2,2),(4,1,1),(3,2,5))}
    for ndim in (2,3):
        for dshape in combinations([19,31,43],ndim):
            for sigma in combinations_with_replacement([3,4,5],ndim):
                for dtype in (np.float32,np.uint16, np.int32):
                    for strides in stridess[ndim]:
                        _test_single(dshape,sigma, dtype = dtype, strides=strides)

    
if __name__ == '__main__':
    # x,y = _test_single((10,10,10),(1,1,2), strides=(1,1,1), dtype = np.uint16, skip_assert=True)
    np.random.seed(31)
    x,y = _test_single((19, 31, 43),(3,3,0), strides=(1,1,1), dtype = np.uint16, skip_assert=True)

    # ind = np.unravel_index(np.argmax(np.abs(1.*x-y)), x.shape)
    # print(ind)

    # print(x[tuple(ind)])
    # print(y[tuple(ind)])
    

    # from gputools import get_device
    # get_device().queue.finish()
