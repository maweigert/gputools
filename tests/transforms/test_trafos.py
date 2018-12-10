""" some image manipulation functions like scaling, rotating, etc...

"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np

from gputools import scale, rotate, shift, affine
from scipy import ndimage


def create_shape(shape):
    d = np.zeros(shape,np.float32)
    ss = tuple([slice(s//10,9*s//10) for s in shape])
    d[ss] = 2+np.random.uniform(0,1,d[ss].shape)
    #d[ss] = 2

    return d


def check_error(func):
    def test_func(check = True, nstacks = 1):
        np.random.seed(42)
        for _ in range(nstacks):
            shape = np.random.randint(22,55,3)
            x = create_shape(shape)
            out1, out2 = func(x)
            if check:
                np.testing.assert_allclose(out1, out2, atol=1e-3, rtol = 1.e-3)
        return x, out1, out2
    return test_func

@check_error
def test_scale(x):
    s = np.random.uniform(.5,1.5,3)
    out1 = scale(x, s, interpolation="nearest")
    out2 = ndimage.zoom(x, s, order=0, prefilter=False)
    return out1, out2

@check_error
def test_shift(x):
    s = 10*np.random.uniform(-1, 1, 3)
    out1 = shift(x, s, interpolation="nearest")
    out2 = ndimage.shift(x, s, order=0, prefilter=False)
    return out1, out2

@check_error
def test_affine(x):
    M = np.eye(4)
    np.random.seed(42)
    M[:3] += .1 * np.random.uniform(-1, 1, (3, 4))
    out1 = affine(x,M, interpolation = "nearest")
    out2 = ndimage.affine_transform(x, M[:3, :3], order=0, prefilter=False)
    return out1,out2



def test_rotate():
    d = create_shape((101,102,103))
    out = rotate(d,(1,1,1), angle = .4, center =  (10,10,10), interpolation ="linear")
    return out

#
# def test_scale():
#     d = create_shape((101,102,103))
#     scale(d,1.7, interpolation = "linear")
#     out = scale(d,.3, interpolation  = "nearest")
#     return out

#
# def test_shift(check = True):
#     x = create_shape((101,102,103))
#     dx = (10,10,10)
#     out1 = shift(x,dx, interpolation = "nearest")
#     out2 = ndimage.shift(x,dx,order=0)
#     if check:
#         np.testing.assert_allclose(out1,out2,atol=1e-5)
#     return out1,out2

#
# def test_affine(check = True):
#     x = create_shape((11,22,33))
#     M = np.eye(4)
#     np.random.seed(42)
#     M[:3] += .1 * np.random.uniform(-1, 1, (3, 4))
#     out1 = affine(x,M, interpolation = "nearest")
#     out2 = ndimage.affine_transform(x, M[:3, :3], order=0)
#     #out2 = ndimage.affine_transform(x,np.linalg.inv(M[:3,:3]),order=1)
#
#     if check:
#         np.testing.assert_allclose(out1,out2,atol=1e-1, rtol = 1e-1)
#     return out1,out2


def test_modes():
    d = create_shape((101,101,101))
    outs = []
    for mode in ("constant","edge","wrap"):
        for interp in ("linear", "nearest"):
            print(interp, mode)
            outs.append(rotate(d,axis = (0,1,0), angle = 0.4, mode = mode, interpolation=interp))
    return outs

if __name__ == '__main__':
    
    pass
    # out1 = test_scale()
    # out2 =  test_rotate()
    # out3 =  test_shift()
    # outs = test_modes()
    # outs = test_affine()


    
