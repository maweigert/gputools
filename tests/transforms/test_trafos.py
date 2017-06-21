""" some image manipulation functions like scaling, rotating, etc...

"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np

from gputools import scale, rotate, translate, affine


def create_shape(shape):
    d = np.ones(shape,np.float32)
    ss = tuple([slice(s//10,9*s//10) for s in shape])
    d[ss] = 2.
    return d

def test_scale():
    d = create_shape((101,102,103))
    scale(d,1.7, interpolation = "linear")
    out = scale(d,.3, interpolation  = "nearest")
    return out

def test_rotate():
    d = create_shape((101,102,103))
    out = rotate(d,(1,1,1), angle = .4, center =  (10,10,10), interpolation ="linear")
    return out

def test_translate():
    d = create_shape((101,102,103))
    out = translate(d,(10,10,10), interpolation = "nearest")
    return out

def test_modes():
    d = create_shape((101,101,101))
    outs = []
    for mode in ("constant","edge","wrap"):
        for interp in ("linear", "nearest"):
            print(interp, mode)
            outs.append(rotate(d,axis = (0,1,0), angle = 0.4, mode = mode, interpolation=interp))
    return outs

if __name__ == '__main__':
    

    out1 = test_scale()
    out2 =  test_rotate()
    out3 =  test_translate()
    outs = test_modes()


    
