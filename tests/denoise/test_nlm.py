from __future__ import print_function, unicode_literals, absolute_import, division
from gputools.denoise import  nlm2, nlm3


import numpy as np


def test_2d(fac = .3):
    from scipy.misc import ascent

    d = ascent().astype(np.float32)

    d = d[:,:-10]
    sig = .2*np.amax(d)

    y = d+sig*np.random.uniform(0,1.,d.shape)

    out = nlm2(y.astype(np.float32),fac*sig,3,5)

    return y, out

def test_3d(fac = .3):

    x = np.linspace(-1,1,100)
    R = np.sqrt(np.sum([X**2 for X in np.meshgrid(x,x,x,indexing="ij")],axis=0))
    d = 1.*(R<.4)

    sig = .2*np.amax(d)

    y = d+sig*np.random.uniform(0,1.,d.shape)

    out = nlm3(y.astype(np.float32),fac*sig,3,5)
    return y, out


if __name__ == '__main__':


    y2, out2 = test_2d()
    #y3, out3 = test_3d(10)
