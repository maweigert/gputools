"""


mweigert@mpi-cbg.de

"""
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
    print "shape = %s, axes = %s, dtype = %s"%(data.shape, axes, data.dtype) + colored("\t[OK]","blue" if is_equal else "red")
    assert is_equal

def test_all():
    from itertools import combinations

    for ndim in xrange(1,4):
        d = create_data([140+20*n for n in xrange(ndim)])
        for n_ax in xrange(1,min(ndim,3)+1):
            for axes in combinations(range(ndim),n_ax):
                check_single(d, axes)
                check_single(d.astype(np.complex64), axes)


if __name__ == '__main__':


    test_all()