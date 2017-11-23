""" some image manipulation functions like scaling, rotating, etc...

"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np

from gputools import scale
from gputools.transforms.scale import scale_bicubic

def create_shape(shape):
    d = np.ones(shape,np.float32)
    ss = tuple([slice(s//10,9*s//10) for s in shape])
    d[ss] = 2.
    return d

def test_scale():
    d = create_shape((21,23,44))
    out = scale(d,1.5, interpolation  = "nearest")
    return out

def test_scale_bicubic():
    d = create_shape((21, 23, 44))
    out = scale_bicubic(d, 1.5)
    return out


if __name__ == '__main__':
    

    out1 = test_scale()
    out2 = test_scale_bicubic()


    
