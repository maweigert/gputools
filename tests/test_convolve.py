import numpy as np
from time import time
from scipy.signal import convolve2d

import gputools


def test_convolve2():
    pass

if __name__ == '__main__':
    test_convolve2()

    from scipy.misc import lena
    
    # N = 256
    # d = np.random.uniform(-1,1,(N,)*2)
    # h = np.random.uniform(-1,1,(N,)*2)

    d = lena()[100:164,100:164]

    x = np.linspace(-1,1,65)[:-1]
    Y,X = np.meshgrid(x,x,indexing="ij")
    
    h = np.exp(-1000*(Y**2+X**2))
    h *= 1./np.sum(h)
    
    t = time()
    res1 = convolve2d(d,h,mode="same")
    print time()-t

    t = time()    
    res2 = gputools.fft_convolve(d,h)
    print time()-t
    
    
