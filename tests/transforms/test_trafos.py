""" some image manipulation functions like scaling, rotating, etc...

"""

import numpy as np

from gputools import scale, rotate, translate


def create_shape(shape):
    d = np.zeros(shape,np.float32)
    ss = tuple([slice(s/4,3*s/4) for s in shape])
    d[ss] = 1.
    return d

def test_scale():
    d = create_shape((101,102,103))
    scale(d,1.7, interp = "linear")
    out = scale(d,.3, interp = "nearest")
    return out

def test_rotate():
    d = create_shape((101,102,103))
    out = rotate(d, (10,10,10), (1,1,1), angle = .4, mode="linear")
    return out

def test_translate():
    d = create_shape((101,102,103))
    out = translate(d,10,10,10, mode = "nearest")
    return out

if __name__ == '__main__':
    

    out1 = test_scale()
    out2 =  test_rotate()
    out3 =  test_translate()


    
