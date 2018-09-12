""" some image manipulation functions like scaling, rotating, etc...

"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np

from gputools import scale
from gputools.transforms.scale import scale_bicubic

from scipy.ndimage.interpolation import zoom

np.random.seed(0)

def create_shape(shape):
    d = np.ones(shape,np.float32)
    ss = tuple([slice(s//10,9*s//10) for s in shape])
    d[ss] = np.random.uniform(0,1,d[ss].shape)
    return d

def scale_func(x,zoom_factor):
    res_gputools = scale(x,zoom_factor, interpolation="linear")
    res_scipy = zoom(x,zoom_factor, order=1, prefilter=False)
    return res_gputools, res_scipy


def test_scale():
    d = create_shape((21,23,44))
    res_gputools, res_scipy = scale_func(d, (1.5,1,1))
    return res_gputools, res_scipy

# def test_scale_bicubic():
#     d = create_shape((21, 23, 44))
#     out = scale_bicubic(d, 1.5)
#     return out


if __name__ == '__main__':
    

    out1, out2 = test_scale()
    # out2 = test_scale_bicubic()


    
