import numpy as np
import numpy.testing as npt
from time import time
import scipy.ndimage.filters as sp_filter

import gputools


def test_convolve2():
    np.random.seed(1)
    N = 256
    Nh = 11 
    d = np.random.uniform(-1,1,(N,)*2).astype(np.float32)
    h = np.random.uniform(-1,1,(Nh,)*2).astype(np.float32)
    
    out1 = sp_filter.convolve(d,h,mode="constant")

    out2 = gputools.convolve(d,h)

    npt.assert_allclose(out1,out2,atol=1.e-5)



def test_convolve1():

    np.random.seed(1)
    N = 256
    Nh = 11 
    d = np.random.uniform(-1,1,(N,)*1).astype(np.float32)
    h = np.random.uniform(-1,1,(Nh,)*1).astype(np.float32)
    out1 = sp_filter.convolve(d,h,mode="constant")

    out2 = gputools.convolve(d,h)

    npt.assert_allclose(out1,out2,atol=1.e-5)

    
if __name__ == '__main__':
    # test_convolve2()

   
    np.random.seed(1)
    N = 64
    Nh = 7 
    d = np.random.uniform(-1,1,(N,)*3).astype(np.float32)
    h = np.random.uniform(-1,1,(Nh,)*3).astype(np.float32)
    
    out1 = sp_filter.convolve(d,h,mode="constant")

    out2 = gputools.convolve(d,h)

    npt.assert_allclose(out1,out2,rtol=1.e-2,atol=1.e-5)

