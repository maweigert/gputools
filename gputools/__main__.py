
import gputools
import numpy as np


def _convolve_rand(dshape,hshape):
    print "convolving test: dshape = %s, hshape  = %s"%(dshape,hshape)
    np.random.seed(1)
    d = np.random.uniform(-1,1,dshape).astype(np.float32)
    h = np.random.uniform(-1,1,hshape).astype(np.float32)
    
    out2 = gputools.convolve(d,h)

    
def test_convolve():
    for ndim in [1,2,3]:
        for N in range(10,200,40):
            for Nh in range(3,11,2):
                dshape = [N/ndim+3*n for n in range(ndim)]
                hshape = [Nh+3*n for n in range(ndim)]
                
                _convolve_rand(dshape,hshape)
    
if __name__ == '__main__':
    test_convolve()
