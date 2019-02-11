import numpy as np
import scipy.ndimage.filters as spf
from scipy import ndimage
from skimage.restoration import denoise_nl_means
from skimage.transform import integral_image as sk_integral_image
from gputools.convolve import median_filter, uniform_filter, gaussian_filter
from gputools.transforms import scale, integral_image
from gputools.denoise import nlm3
from gputools import fft
from gputools import OCLArray, get_device


from time import time


type_name_dict = {
    np.uint8:"uint8",
    np.uint16:"uint16",
    np.float32:"float32",
    np.complex64:"complex64",
}
    

def bench(description, dshape, dtype, func_cpu, func_gpu, func_gpu_notransfer=None, niter=2):
    x = np.random.randint(0,100,dshape).astype(dtype)

    func_cpu(x)
    t_cpu = time()
    for _ in range(niter):
        y = func_cpu(x);
    t_cpu = (time()-t_cpu)/niter

    func_gpu(x)
    t_gpu = time()
    for _ in range(niter):
        y = func_gpu(x);
    t_gpu = (time() - t_gpu)/niter

    if func_gpu_notransfer is not None:
        x_g  = OCLArray.from_array(x)
        tmp_g  = OCLArray.empty_like(x)
        func_gpu_notransfer(x_g, tmp_g);
        get_device().queue.finish()
        t_gpu_notransfer = time()
        for _ in range(niter):
            func_gpu_notransfer(x_g, tmp_g);
        get_device().queue.finish()
        t_gpu_notransfer = (time() - t_gpu_notransfer)/niter
    else:
        t_gpu_notransfer = None
    # print("%s\t\t %s\t%d ms \t %d ms"%(description,dshape, 1000*t1,1000*t2))

    print("%s| %s %s | %d ms | %d ms | %s"%(description,dshape, type_name_dict[dtype],1000*t_cpu,1000*t_gpu, "%d ms"%(1000*t_gpu_notransfer) if t_gpu_notransfer is not None else "-"))
    
    return t_cpu, t_gpu, t_gpu_notransfer



if __name__ == '__main__':

    factor = 1
    cut = lambda s: tuple(_s//factor for _s in s)

    dshape = (128,1024,1024)

    print("description  | dshape |  dtype | t_cpu (ms) | t_gpu (ms) | t_gpu_notrans (ms) ")

    # bench("Mean filter 7x7x7",cut(dshape),np.uint8,
    #       lambda x: spf.uniform_filter(x, 7),
    #       lambda x: uniform_filter(x, 7),
    #       lambda x_g, res_g: uniform_filter(x_g, 7, res_g = res_g)
    #       )
    #
    # bench("Median filter 3x3x3",cut(dshape),np.uint8,
    #       lambda x: spf.median_filter(x,size = 3),
    #       lambda x: median_filter(x, size=3),
    #       lambda x_g, res_g: median_filter(x_g, size=3, res_g = res_g)
    #       )
    #
    # bench("Gaussian filter 5x5x5",cut(dshape),np.float32,
    #       lambda x: spf.gaussian_filter(x, 5),
    #       lambda x: gaussian_filter(x, 5),
    #       lambda x_g, res_g: gaussian_filter(x_g, 5, res_g = res_g)
    #       )
    #
    # bench("Zoom/Scale 2x2x2",cut(dshape),np.uint16,
    #       lambda x: ndimage.zoom(x,(2,)*3, order=1, prefilter=False),
    #       lambda x: scale(x, (2,)*3, interpolation="linear")
    #       )
    #
    #
    # bench("NLM denoising",cut((64,256,256,)),np.float32,
    #       lambda x: denoise_nl_means(x,5,5,multichannel=False),
    #       lambda x: nlm3(x,.1,2,5),
    #       )
    #
    # bench("FFT",cut(dshape),np.complex64,
    #       lambda x: np.fft.fftn(x),
    #       lambda x: fft(x),
    #       lambda x_g, res_g: fft(x_g, inplace = True)
    #       )
    #

    bench("Integral Image",cut((512,512,256,)),np.float32,
          lambda x: sk_integral_image(x),
          lambda x: integral_image(x),
          lambda x_g, res_g: integral_image(x_g, res_g = res_g)
          )


