""" Lucy richardson deconvolution
"""
from __future__ import print_function, unicode_literals, absolute_import, division

import logging
logger = logging.getLogger(__name__)

import os
import numpy as np
from gputools.deconv import deconv_rl
from gputools.convolve import convolve

def test_2d():
    N = 512
    d = np.zeros((N,)*2)

    ind = np.random.randint(10,N-10,(2,100))
    d[ind[0],ind[1]] = 1.
    
    h = np.ones((11,)*2)/121.


    y = convolve(d,h)

    y += 0.0*np.amax(d)*np.random.uniform(0,1,d.shape)

    print("start")
    
    #u = deconv_rl(y,h, 2)


if __name__ == '__main__':

    N = 128
    d = np.zeros((N,)*3)

    ind = np.random.randint(10,N-10,(3,100))
    d[ind[0],ind[1],ind[2]] = 1.
    
    hx = np.exp(-10*np.linspace(-1,1,5)**2)
    h = np.einsum("i,j,k",hx,hx,hx)

    # h = np.ones((11,)*3)/121.
    y = convolve(d,h)

    y += 0.0*np.amax(d)*np.random.uniform(0,1,d.shape)

    print("start")
    
    u = deconv_rl(y.astype(np.float32),h.astype(np.float32), 2)


