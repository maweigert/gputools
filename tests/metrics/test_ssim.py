"""

mweigert@mpi-cbg.de
"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import numpy.testing as npt
from skimage.metrics import structural_similarity
from itertools import combinations_with_replacement
from time import time
from gputools.metrics import ssim

def _time_me(func, dshape, niter = 2):
    d0 = np.random.uniform(-1,1,dshape).astype(np.float32)
    drange = np.amax(d0) - np.amin(d0)
    t = time()
    for _ in range(niter):
        func(d0, d0, data_range=drange)
    return (time()-t)/niter

def time_cpu(dshape, niter=2):
    return _time_me(compare_ssim, dshape, niter)

def time_gpu(dshape, niter=2):
    return _time_me(ssim, dshape, niter)


def _test_single(dshape):
    d0 =np.zeros(dshape,np.float32)

    ss = tuple(slice(s//4,-s//4) for s in dshape)
    d0[ss] = 10.

    d1 = d0+np.random.uniform(0, 1, dshape).astype(np.float32)

    drange = np.amax(d0) - np.amin(d0)
    print(drange)

    m1, m2 = structural_similarity(d0,d1, data_range = drange), ssim(d0,d1)

    print("shape: %s \t\tSSIM = %.2f\t\tdifference: %s" % (dshape, m1, np.abs(m1-m2)))
    npt.assert_almost_equal(m1,m2, decimal = 5)
    return m1, m2


def test_acc():
    for ndim in [2, 3]:
        for dshape in combinations_with_replacement([40, 50, 60], ndim):
            _test_single(dshape)

def get_times():

    ns = 2**np.arange(5,13)

    ts1 = [time_cpu((n,)*2,2) for n in ns]
    ts2 = [time_gpu((n,)*2,2) for n in ns]
    return ns, ts1, ts2




if __name__ == '__main__':

    test_acc()
    #
    # try:
    #     print(time_gpu((512,)*3))
    # except Exception as e:
    #     print("catched:", e)
