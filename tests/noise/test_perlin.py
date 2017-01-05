""" Lucy richardson deconvolution
"""
from __future__ import print_function, unicode_literals, absolute_import, division
import logging
logger = logging.getLogger(__name__)


import numpy as np
from gputools.noise import perlin2, perlin3

def test_2d(size = (100,103)):

    y = perlin2(size=size, units=(.5,.6), repeat=(3,5))

    return y

def test_3d(size = (100,103,105)):
    y = perlin3(size=size, units=(.5,.6,.7), repeat=(3,4,5))
    return y


if __name__ == '__main__':

    y1 = test_2d()
    y2 = test_3d()