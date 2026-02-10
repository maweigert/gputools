
import os
import numpy as np
import numpy.testing as npt
import pytest
from itertools import product, combinations
from gputools import fft,  fft_plan

_IS_MACOS_CI = os.environ.get("CI") and os.sys.platform == "darwin"


def _single_batched(d, axes):
    res1 = np.fft.fftn(d, axes=axes)
    res2 = fft(d, axes=axes)
    return res1, res2


@pytest.mark.skipif(_IS_MACOS_CI, reason="reikna FFT incorrect on pocl CPU device")
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
