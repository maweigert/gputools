import numpy as np
from numpy import *

from scipy.signal import convolve2d
              
from skimage import color, data, restoration

from time import time
from gputools.deconv.deconv_rl import _deconv_rl_gpu_conv as _deconv_g
import gputools

def time_wrap(f):
    def func_to_time(im_size = 256, psf_size = 5, Niter = 10, is_pad = False ):
        d = zeros((im_size,)*2,np.float32)
        if is_pad:
            h = ones((im_size,)*2,np.float32)
        else:
            h = ones((psf_size,)*2,np.float32)

        t = time()
        for _ in range(Niter):
            f(d, h)
        return (time()-t)/Niter

    return func_to_time

def time_wrap_g(f):
    def func_to_time(im_size = 256, psf_size = 5, Niter = 10, is_pad = False ):
        d = zeros((im_size,)*2,np.float32)
        if is_pad:
            h = ones((im_size,)*2,np.float32)
        else:
            h = ones((psf_size,)*2,np.float32)
        d_g = gputools.OCLArray.from_array(d)
        h_g = gputools.OCLArray.from_array(h)
        gputools.get_device().queue.finish()

        t = time()
        for _ in range(Niter):
            f(d_g, h_g)
        
        gputools.get_device().queue.finish()

        return (time()-t)/Niter

    return func_to_time

@time_wrap
def conv_scikit(d,h):
    return convolve2d(d, h, 'same')

@time_wrap
def conv_gpu(d,h):
    return gputools.convolve(d, h)

@time_wrap
def deconv_scikit(d,h):
    return restoration.richardson_lucy(d, h, 10, clip = False)


@time_wrap_g
def deconv_gpu(d,h):
    return _deconv_g(d, h, 10)



if __name__ == "__main__":

    ns = 2*arange(2,11)+1
    t1 = [deconv_scikit(64,n,2) for n in ns]
    t2 = [deconv_gpu(256,n,10) for n in ns]


    #t1 = [time_conv_scikit(256,n) for n in ns]
    # t2 = [time_conv_gpu(256,n) for n in ns]
    # t3 = [time_conv_gpu2(256,n) for n in ns]

 
# camera0 = color.rgb2gray(data.camera())
# psf = np.ones((11, 11)) / 21**2
# camera = convolve2d(camera0, psf, 'same')
# camera += 0.01 * camera.std() * np.random.standard_normal(camera.shape)
# out = restoration.richardson_lucy(camera, psf, 5, clip = False)
