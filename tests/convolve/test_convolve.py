from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import numpy.testing as npt
from time import time
import scipy.ndimage.filters as sp_filter
import gputools



def _convolve_rand(dshape, hshape, assert_close = True, test_subblocks = True):
    print("convolving test: dshape = %s, hshape  = %s" % (dshape, hshape))
    np.random.seed(1)
    d = np.random.uniform(-1, 1, dshape).astype(np.float32)
    h = np.random.uniform(-1, 1, hshape).astype(np.float32)


    print("gputools")
    outs = [gputools.convolve(d, h)]

    print("scipy")
    out1 = sp_filter.convolve(d, h, mode="constant", cval = 0.)

    if test_subblocks:
        outs.append(gputools.convolve(d, h, sub_blocks=(2, 3, 4)))

    if assert_close:
        for out in outs:
            npt.assert_allclose(out1, out, rtol=1.e-2, atol=1.e-3)

    return [out1]+outs

def test_convolve():
    for ndim in [1, 2, 3]:
        for N in range(10, 200, 40):
            for Nh in range(3, 11, 2):
                dshape = [N // ndim + 3 * n for n in range(ndim)]
                hshape = [Nh + 3 * n for n in range(ndim)]

                _convolve_rand(dshape, hshape)


def test_reflect():
    image = np.ones((5, 5))
    image[2, 2] = 0
    h = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    out = gputools.convolve(image, h, mode='reflect')
    npt.assert_allclose(out[0], [0,] * 5)
    npt.assert_allclose(out[1], [0, 1, 2, 1, 0])


def test_small():
    # for N1 in range(10, 40, 7):  # <-- out of resources on macbook pro
    for N1 in range(10, 25, 5):
        _convolve_rand((N1,) * 3, (N1,) * 3, test_subblocks=False)

if __name__ == '__main__':
    test_small()
    # a = _convolve_rand((31,)*3,(31,)*3, assert_close=False, test_subblocks=False)

