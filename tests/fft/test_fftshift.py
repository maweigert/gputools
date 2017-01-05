"""


mweigert@mpi-cbg.de

"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from termcolor import colored
from gputools import OCLArray, fftshift

def create_data(dshape, use_complex= False):
    d = np.random.uniform(-1,1,dshape)
    if use_complex:
        d = d+.4j*d
    return d

def check_single(data, axes = None):
    is_equal = np.allclose(np.fft.fftshift(data,axes = axes),fftshift(data, axes = axes))
    print("shape = %s, axes = %s, dtype = %s"%(data.shape, axes, data.dtype) + colored("\t[OK]","blue" if is_equal else "red"))
    assert is_equal

def test_all():
    from itertools import combinations

    for ndim in range(1,4):
        d = create_data([140+20*n for n in range(ndim)])
        for n_ax in range(1,min(ndim,3)+1):
            for axes in combinations(list(range(ndim)),n_ax):
                check_single(d, axes)
                check_single(d.astype(np.complex64), axes)


if __name__ == '__main__':


    test_all()