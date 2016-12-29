
import numpy as np
import numpy.testing as npt
import gputools

from gputools.convolve import  max_filter, min_filter

from scipy.ndimage.filters import maximum_filter, minimum_filter


np.random.seed(10)


def _test_single_minmax(dshape, size):
    d = np.random.uniform(-1, 1., dshape).astype(np.float32)

    out1 = maximum_filter(d, size)
    out2 = max_filter(d, size)
    print(("shape: %s \t\tsize: %s\t\tdifference: %s" % (dshape, size,np.amax(np.abs(out1 - out2)))))
    npt.assert_almost_equal(out1,out2)
    return out1, out2


def test_all():
    from itertools import product, combinations_with_replacement


    for ndim in [2,3]:
        for dshape in combinations_with_replacement([40,50,60],ndim):
            for size in [3,7,13]:
                _test_single_minmax(dshape,size)


if __name__ == '__main__':

    test_all()