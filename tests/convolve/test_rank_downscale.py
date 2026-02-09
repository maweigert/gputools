import numpy as np
import numpy.testing as npt
from itertools import combinations, combinations_with_replacement
from gputools.convolve import rank_downscale

from skimage.measure import block_reduce

def _test_func(dshape, size , dtype = np.float32, func=np.median, rank=None, skip_assert = False):
    d = np.random.randint(0, 40, dshape).astype(dtype)

    out1 = block_reduce(d, size, func)
    out2 = rank_downscale(d, size, rank=rank)

    print(("shape: %s \tsize: %s\t dtype: %s\tdifference: %.2f" % (dshape, size, dtype,np.amax(np.abs(out1 - out2)))))
    if not skip_assert:
        npt.assert_almost_equal(out1,out2, decimal = 5)
    return out1, out2


def test_all():
    for ndim in [2,3]:
        for dshape in combinations([30,60],ndim):
            for size in combinations_with_replacement([3, 5], ndim):
                    for dtype in (np.uint8, np.uint16, np.float32):
                        for func, rank in zip((np.median, np.min, np.max), (None, 0, -1)):
                            _test_func(dshape,size, dtype=dtype, func=func, rank=rank)

np.random.seed(0)

if __name__ == '__main__':

    test_all()
    # _test_median((30,30), 3)