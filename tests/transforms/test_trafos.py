""" some image manipulation functions like scaling, rotating, etc...

"""

import numpy as np

from gputools import scale, rotate, translate



def test_scale():
    d = np.zeros((100,100,100),np.float32)
    scale(d,1.7, interp = "linear")
    scale(d,.3, interp = "linear")

def test_rotate():
    d = np.zeros((100,100,100),np.float32)
    rotate(d,(10,10,10),(1,1,1),angle = .4, interp = "linear")

def test_translate():
    d = np.zeros((100,100,100),np.float32)
    translate(d,10,10,10, interp = "nearest")

if __name__ == '__main__':
    

    test_scale()
    test_rotate()
    test_translate()
    
    
