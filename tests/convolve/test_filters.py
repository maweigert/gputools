from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import numpy.testing as npt
from itertools import product, combinations_with_replacement
import scipy.ndimage.filters as spf
import gputools

from gputools.convolve.filters import  max_filter, min_filter, uniform_filter


np.random.seed(0)


def _test_single(filt1, filt2, dshape, size , cval = 0., skip_assert = False):
    d = np.random.uniform(-1, 1., dshape).astype(np.float32)

    out1 = filt1(d, size)
    out2 = filt2(d, size, mode = "constant", cval = cval)
    print(("shape: %s \t\tsize: %s\t\tdifference: %s" % (dshape, size,np.amax(np.abs(out1 - out2)))))
    if not skip_assert:
        npt.assert_almost_equal(out1,out2, decimal = 5)
    return out1, out2


def _test_some(filt1, filt2, cval = 0.):
    for ndim in [2,3]:
        for dshape in combinations_with_replacement([40,50,60],ndim):
            for size in [3,7,13]:
                _test_single(filt1, filt2, dshape,size, cval = cval)

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
