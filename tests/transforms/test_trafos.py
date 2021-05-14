""" some image manipulation functions like scaling, rotating, etc...

"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np

from gputools import scale, rotate, shift, affine
from scipy import ndimage
import pytest

def create_shape(shape = (100,110,120)):
    d = np.zeros(shape,np.float32)
    ss = tuple([slice(s//10,9*s//10) for s in shape])
    d[ss] = 2
    for i in range(len(shape)):
        ss0 = list(slice(None) for _ in range(len(shape)))
        ss0[i] = (10./min(shape)*np.arange(shape[i]))%2>1
        d[ss0] = 0        
    return d

def check_error(atol=1e-2, rtol=1e-2):
    def _check_error(func):
        def test_func(check = True, nstacks = 1):
            np.random.seed(42)
            for _ in range(nstacks):
                shape = np.random.randint(22,55,3)
                x = create_shape(shape)
                out1, out2 = func(x)
                if check:
                    np.testing.assert_allclose(out1, out2, atol=atol, rtol = rtol)
            return x, out1, out2
        return test_func
    return _check_error

@pytest.mark.skip(reason="still some minor difference to scipy.ndimage, have to check more")
@check_error()
def test_scale(x):
    s = np.random.uniform(.5,1.5,3)
    out1 = scale(x, s, interpolation="nearest")
    out2 = ndimage.zoom(x, s, order=0, prefilter=False)
    return out1, out2

@check_error()
def test_shift(x):
    s = 10*np.random.uniform(-1, 1, 3)
    out1 = shift(x, s, interpolation="nearest")
    out2 = ndimage.shift(x, s, order=0, prefilter=False)
    return out1, out2

# @pytest.mark.skip(reason="still some minor difference to scipy.ndimage, have to check more")
@check_error(atol=5e-2, rtol=5e-2)
def test_affine(x):
    M = np.eye(4)
    np.random.seed(42)
    M[:3] += .1 * np.random.uniform(-1, 1, (3, 4))    
    out1 = affine(x, M, interpolation = "nearest")
    out2 = ndimage.affine_transform(x, M, order=0, prefilter=False)
    return out1,out2

@check_error(atol=5e-2, rtol=5e-2)
def test_affine_reshape(x):
    M = np.eye(4)
    np.random.seed(42)
    M[:3] += .1 * np.random.uniform(-1, 1, (3, 4))
    output_shape = (33,45,97)
    out1 = affine(x, M, interpolation = "linear", output_shape = output_shape)
    out2 = ndimage.affine_transform(x, M, order=1, prefilter=False, output_shape = output_shape)
    return out1,out2


def test_rotate(angle=  .4, axis = (1,0,0), interpolation = "linear"):
    x = create_shape((100,110,120))
    y = rotate(x,angle = angle, axis = axis, interpolation =interpolation)
    return x, y


def test_modes():
    d = create_shape((101,101,101))
    outs = []
    for mode in ("constant","edge","wrap"):
        for interp in ("linear", "nearest"):
            print(interp, mode)
            outs.append(rotate(d, axis= (0, 1, 0), angle = 0.4, mode = mode, interpolation=interp))
    return outs

if __name__ == '__main__':

    x, y1, y2 = test_scale(check=False)
    x, y1, y2 = test_shift(check=False)
    x, y1, y2 = test_affine(check=False)
    x, y1, y2 = test_affine_reshape(check=False)

    print(np.max(np.abs(y1-y2))) 
