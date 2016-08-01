"""


mweigert@mpi-cbg.de

"""

import numpy as np
from gputools import OCLArray, fft, get_device, fft_plan
from time import time

def time_gpu(dshape, niter =100, fast_math = False):
    d_g = OCLArray.empty(dshape, np.complex64)
    get_device().queue.finish()
    plan = fft_plan(dshape, fast_math = fast_math)
    t = time()
    for _ in xrange(niter):
        fft(d_g, inplace=True, plan = plan)
    get_device().queue.finish()
    t = (time()-t)/niter
    print "GPU (fast_math = %s)\t%s\t\t%.2f ms"%(fast_math, dshape, 1000.*t)

def time_np(dshape, niter =20):
    d =np.empty(dshape, np.complex64)
    get_device().queue.finish()
    t = time()
    for _ in xrange(niter):
        np.fft.fftn(d)
    get_device().queue.finish()
    t = (time()-t)/niter
    print "CPU\t\t\t%s\t\t%.2f ms"%(dshape, 1000.*t)

def test_speed():
    Ns  =(256,512, 1024, 2048)

    for N in Ns:
        time_gpu((N,)*2, fast_math=True)
        time_gpu((N,)*2, fast_math=False)
        time_np((N,)*2)

if __name__ == '__main__':


    test_speed()