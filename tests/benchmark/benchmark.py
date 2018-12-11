import numpy as np
import scipy.ndimage.filters as spf
from scipy import ndimage
from skimage.restoration import denoise_nl_means
from gputools.convolve import median_filter, uniform_filter, gaussian_filter
from gputools.transforms import scale
from gputools.denoise import nlm3
from gputools import fft
from gputools import OCLArray, get_device


from time import time


def bench(description, dshape, dtype, func1, func2, func3=None, niter=2):
    x = np.random.randint(0,100,dshape).astype(dtype)

    func1(x)
    t1 = time()
    for _ in range(niter):
        y = func1(x);
    t1 = (time()-t1)/niter

    func2(x)
    t2 = time()
    for _ in range(niter):
        y = func2(x);
    t2 = (time() - t2)/niter

    if func3 is not None:
        x_g  = OCLArray.from_array(x)
        tmp_g  = OCLArray.empty_like(x)
        func3(x_g, tmp_g);
        get_device().queue.finish()
        t3 = time()
        for _ in range(niter):
            func3(x_g, tmp_g);
        get_device().queue.finish()
        t3 = (time() - t3)/niter
    else:
        t3 = None
    # print("%s\t\t %s\t%d ms \t %d ms"%(description,dshape, 1000*t1,1000*t2))

    print("%s| %s| %d ms | %d ms | %s"%(description,dshape, 1000*t1,1000*t2, "%d ms"%(1000*t3) if func3 is not None else "-"))
    
    return t1, t2



if __name__ == '__main__':

    factor = 1
    cut = lambda s: tuple(_s//factor for _s in s)

    dshape = (128,1024,1024)
    
    # bench("Median filter 3x3x3",cut(dshape),np.uint8,
    #       lambda x: spf.median_filter(x,size = 3),
    #       lambda x: median_filter(x, size=3),
    #       lambda x_g, res_g: median_filter(x_g, size=3, res_g = res_g)
    #       )

    # bench("Gaussian filter 5x5x5",cut(dshape),np.float32,
    #       lambda x: spf.gaussian_filter(x, 5),
    #       lambda x: gaussian_filter(x, 5),
    #       lambda x_g, res_g: gaussian_filter(x_g, 5, res_g = res_g)
    #       )

    # bench("Zoom/Scale 2x2x2",cut(dshape),np.uint8,
    #       lambda x: ndimage.zoom(x,(2,)*3, order=1, prefilter=False),
    #       lambda x: scale(x, (2,)*3, interpolation="linear")
    #       )


    bench("NLM denoising",cut((64,256,256,)),np.float32,
          lambda x: denoise_nl_means(x,5,5,multichannel=False),
          lambda x: nlm3(x,.1,2,5),
          )

    bench("FFT",cut(dshape),np.complex64,
          lambda x: np.fft.fftn(x),
          lambda x: fft(x),
          lambda x_g, res_g: fft(x_g, inplace = True)
          )


