import numpy as np
import scipy.ndimage.filters as spf
from scipy import ndimage
from gputools.convolve import median_filter, uniform_filter
from gputools.transforms import scale
from skimage.restoration import nl_means_denoising
from gputools.denoise import nlm3

from time import time


def bench(description, dshape, dtype, func1, func2, niter=2):
    x = np.random.randint(0,100,dshape).astype(dtype)

    t1 = time()
    for _ in range(niter):
        y = func1(x);

    t1 = (time()-t1)
    t2 = time()
    for _ in range(niter):
        y = func2(x);
    t2 = (time() - t2)
    print("%d ms \t %d ms"%(1000*t1,1000*t2))
    return t1, t2



if __name__ == '__main__':
    bench("median filter 3x3x3",(128,)*3,np.float32,
          lambda x: spf.median_filter(x,size = (3,3,3)),
          lambda x: median_filter(x, size=(3, 3, 3))
          )

    bench("mean filter 5x5x5", (256,)*3, np.uint8,
          lambda x: spf.uniform_filter(x, size=(5, 5, 5)),
          lambda x: uniform_filter(x, size=(5, 5, 5))
          )

    bench("scale 2x2x2",(128,)*3,np.uint16,
          lambda x: ndimage.zoom(x,(2,)*3, order=1, prefilter=False),
          lambda x: scale(x, (2,)*3, interpolation="linear")
          )




