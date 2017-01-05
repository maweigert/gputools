"""


mweigert@mpi-cbg.de

"""

from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from gputools import OCLArray, fft, get_device, fft_plan
from time import time


def time_gpu(dshape, niter=100, fast_math=False):
    d_g = OCLArray.empty(dshape, np.complex64)
    get_device().queue.finish()
    plan = fft_plan(dshape, fast_math=fast_math)
    t = time()
    for _ in range(niter):
        fft(d_g, inplace=True, plan=plan)
    get_device().queue.finish()
    t = (time() - t) / niter
    print("GPU (fast_math = %s)\t%s\t\t%.2f ms" % (fast_math, dshape, 1000. * t))
    return t


def time_np(dshape, niter=3):
    d = np.empty(dshape, np.complex64)
    get_device().queue.finish()
    t = time()
    for _ in range(niter):
        np.fft.fftn(d)
    get_device().queue.finish()
    t = (time() - t) / niter
    print("CPU\t\t\t%s\t\t%.2f ms" % (dshape, 1000. * t))
    return t


def test_speed():
    Ns = (256, 512, 1024, 2048)

    for N in (256, 512, 1024, 2048):
        t0 = time_np((N,) * 2)
        t1 = time_gpu((N,) * 2, fast_math=True)
        t2 = time_gpu((N,) * 2, fast_math=False)
        print("speedup: %.1f / %.1f " % (t0 / t1, t0 / t2))
        print("-" * 50)

    for N in (128, 256):
        t0 = time_np((N,) * 3)
        t1 = time_gpu((N,) * 3, fast_math=True)
        t2 = time_gpu((N,) * 3, fast_math=False)
        print("speedup: %.1f / %.1f " % (t0 / t1, t0 / t2))
        print("-" * 50)


if __name__ == '__main__':
    test_speed()
