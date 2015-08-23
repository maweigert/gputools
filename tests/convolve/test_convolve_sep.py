
import numpy as np
import gputools


def test_conv_sep():

    N  = 128
    d = np.zeros((N,N+3,N+5),np.float32)

    d[N/2,N/2,N/2]  = 1.

    h = np.exp(-10*np.linspace(-1,1,17)**2)

    res = gputools.convolve_sep3(d,h,h,h)

if __name__ == '__main__':

    pass
    
