"""

mweigert@mpi-cbg.de
"""
from __future__ import print_function, unicode_literals, absolute_import, division

import numpy as np
from gputools.separable import separable_approx, separable_series


from gputools.separable.separable_approx import _separable_series2

def test_2d():
    x = np.linspace(-1, 1, 61)
    y = np.linspace(-1, 1, 101)
    Y,X = np.meshgrid(y,x,indexing = "ij")

    u = np.exp(-3*(2.*X-np.sin(4*Y))**2)
    u += .05 * np.random.uniform(-1, 1, u.shape)

    hs = separable_approx(u, 50)

    print("########" * 5, "separable approximation dim = %s " % u.ndim)
    print("reconstruction error:")
    for i, h in enumerate(hs):
        print("i = %s \t difference: %s"%(i+1,np.amax(np.abs(h-u))/np.amax(np.abs(u))))

    return u, hs

def test_3d():
    x = np.linspace(-1, 1, 31)
    y = np.linspace(-1, 1, 60)
    z = np.linspace(-1, 1, 101)
    Z, Y,X = np.meshgrid(z,y,x,indexing = "ij")

    u = np.exp(-3*(2.*X-np.sin(4*Y)+3.*np.abs(Z))**2)
    u += .08 * np.random.uniform(-1, 1, u.shape)

    hs = separable_approx(u, 100)

    print("########"*5,"separable approximation dim = %s "%u.ndim)
    print("reconstruction error:")
    for i, h in enumerate(hs):
        print("i = %s \t difference: %s"%(i+1,np.amax(np.abs(h-u))/np.amax(np.abs(u))))

    return u, hs





if __name__ == '__main__':

    u2, hs2 = test_2d()
    u3, hs3 = test_3d()