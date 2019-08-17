
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import numpy.testing as npt
from itertools import product, combinations
from termcolor import colored

from gputools import fft,  fft_plan


def _single_batched(d, axes):
    res1 = np.fft.fftn(d, axes=axes)
    res2 = fft(d, axes=axes)
    return res1, res2


def test_batched():
    for ndim in [1, 2, 3]:
        dshape = 1024//(2**np.arange(ndim, 2*ndim))
        d = 1.+np.random.uniform(-1, 1, dshape).astype(np.complex64)

        for n_axes in range(1,ndim+1):
            for axes in combinations(range(0,ndim), n_axes):
                print("validating batched fft of size %s and axes %s" % (dshape, axes))
                res1, res2 = _single_batched(d, axes)
                npt.assert_allclose(res1, res2,  rtol = 1., atol=1.e-1)
            for axes in combinations(range(-1,-ndim,-1), n_axes):
                print("validating batched fft of size %s and axes %s" % (dshape, axes))
                res1, res2 = _single_batched(d, axes)
                npt.assert_allclose(res1, res2,  rtol = 1., atol=1.e-1)


if __name__ == '__main__':
    test_batched()
