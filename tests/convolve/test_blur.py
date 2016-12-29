import numpy as np
import numpy.testing as npt
from time import time

import gputools

def test_blur():
    for ndim in [1,2,3]:
        for N in range(10,200,40):
            for s in np.linspace(1.,5.,10):
                dshape = [N/ndim+3*n for n in range(ndim)]
                size = [s+.2*i for i in range(ndim)]

                print(dshape, size)
                d = np.random.uniform(0,1,dshape)
                
                gputools.blur(d,s)

    
if __name__ == '__main__':
    test_blur()

   

