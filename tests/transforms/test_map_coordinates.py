""" some image manipulation functions like scaling, rotating, etc...

"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np

from gputools import map_coordinates
from scipy import ndimage
import pytest


def create_shape(shape=(100, 110, 120)):
    d = np.zeros(shape, np.float32)
    ss = tuple([slice(s // 10, 9 * s // 10) for s in shape])
    d[ss] = 1+np.random.uniform(0,1,d[ss].shape)

    for i in range(len(shape)):
        ss0 = list(slice(None) for _ in range(len(shape)))
        ss0[i] = (10. / min(shape) * np.arange(shape[i])) % 2 > 1
        d[ss0] = 0
    return d


def check_error(func):
    def test_func(check=True, nstacks=10):
        np.random.seed(42)
        for _ in range(nstacks):
            ndim = np.random.choice((2,3))
            shape = np.random.randint(22, 55, ndim)
            x = create_shape(shape)
            out1, out2 = func(x)
            if check:
                np.testing.assert_allclose(out1, out2, atol=1e-2, rtol=1.e-2)
        return x, out1, out2

    return test_func



@check_error
def test_map_coordinates(x):
    coordinates = np.stack([np.arange(10) ** 2] * x.ndim)
    coordinates = np.random.randint(0,min(x.shape),(x.ndim,100))
    print(coordinates.shape, x.shape)
    out1 = map_coordinates(x, coordinates, interpolation="linear")
    out2 = ndimage.map_coordinates(x, coordinates, order=1, prefilter=False)
    return out1, out2


if __name__ == '__main__':
    x, y1, y2 = test_map_coordinates(check=False, nstacks=1)
