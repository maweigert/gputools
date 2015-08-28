import numpy as np
from numpy import *

from time import time
import gputools

def time_wrap(f):
    def func_to_time(dshape, Niter = 10, **kwargs):
        d = zeros(dshape,np.float32)
        t = time()
        for _ in range(Niter):
            f(d, **kwargs)
        return (time()-t)/Niter

    return func_to_time

def time_wrap_g(f):
    def func_to_time(dshape, Niter = 10, **kwargs):
        d = zeros(dshape,np.complex64)
        d_g = gputools.OCLArray.from_array(d)
        #burn in
        f(d_g,**kwargs)
        
        gputools.get_device().queue.finish()

        t = time()
        for _ in range(Niter):
            f(d_g,**kwargs)
        
        gputools.get_device().queue.finish()

        return (time()-t)/Niter

    return func_to_time


@time_wrap
def fft_np(d):
    return np.fft.fftn(d)

@time_wrap_g
def fft_gpu(d_g):
    return gputools.fft(d_g, inplace = True)

if __name__ == "__main__":

    ns = 2**arange(4,10)
    ndim = 3

    t1 = [fft_np((n,)*ndim,2) for n in ns]
    print("finished t1")

    t2 = [fft_gpu((n,)*ndim,6) for n in ns]

