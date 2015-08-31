import numpy as np
import numpy.testing as npt
from time import time
import scipy.ndimage.filters as sp_filter

import gputools

def _convolve_rand(dshape,hshape):
    print "convolving test: dshape = %s, hshape  = %s"%(dshape,hshape)
    np.random.seed(1)
    d = np.random.uniform(-1,1,dshape).astype(np.float32)
    h = np.random.uniform(-1,1,hshape).astype(np.float32)
    
    out1 = sp_filter.convolve(d,h,mode="constant")

    out2 = gputools.convolve(d,h)

    npt.assert_allclose(out1,out2,rtol=1.e-2,atol=1.e-5)

def test_convolve():
    for ndim in [1,2,3]:
        for N in range(10,200,40):
            for Nh in range(3,11,2):
                dshape = [N/ndim+3*n for n in range(ndim)]
                hshape = [Nh+3*n for n in range(ndim)]
                
                _convolve_rand(dshape,hshape)
    
if __name__ == '__main__':
    test_convolve()

   

