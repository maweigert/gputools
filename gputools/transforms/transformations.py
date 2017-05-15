""" scaling images

"""

from __future__ import print_function, unicode_literals, absolute_import, division
import logging
logger = logging.getLogger(__name__)

import os
import numpy as np
from gputools import OCLArray, OCLImage, OCLProgram, get_device
from gputools import OCLElementwiseKernel

from gputools.utils import mat4_rotate, mat4_translate
from ._abspath import abspath







def affine(data, mat = np.identity(4), mode ="linear"):
    """affine transform data with matrix mat

    """ 

    bop = {"linear":[],"nearest":["-D","USENEAREST"]}

    if not mode in bop:
        raise KeyError("mode = '%s' not defined ,valid: %s"%(mode, list(bop.keys())))
    
    d_im = OCLImage.from_array(data)
    res_g = OCLArray.empty(data.shape,np.float32)
    mat_g = OCLArray.from_array(np.linalg.inv(mat).astype(np.float32,copy=False))

    prog = OCLProgram(abspath("kernels/transformations.cl")
                      , build_options=bop[mode])

    prog.run_kernel("affine",
                    data.shape[::-1],None,
                    d_im,res_g.data,mat_g.data)

    return res_g.get()


def translate(data,x = 0, y = 0,z = 0, mode = "linear"):
    return affine(data,mat4_translate(x,y,z),mode)


def rotate(data, center = None, axis = (1.,0,0), angle = 0., mode ="linear"):
    """
    rotates data around axis by a given angle

    Parameters
    ----------
    data: ndarray
        3d array

    center: tuple or None
        origin of rotation (cz,cy,cx) in pixels
        if None, center is the middle of data
    axis: tuple
        axis = (x,y,z)
    angle: float
    mode: str

    Returns
    -------
    rotated array

    """
    if center is None:
        center = tuple([s//2 for s in data.shape])

    cz, cy , cx  = center
    m = np.dot(mat4_translate(cx,cy,cz),
               np.dot(mat4_rotate(angle,*axis),
                      mat4_translate(-cx,-cy,-cz)))
    return affine(data, m, mode)



if __name__ == '__main__':
    
    d = np.zeros((200,200,200),np.float32)
    d[20:-20,20:-20,20:-20] = 1.

    # res = translate(d, x = 10, y = 5, z= -10 )
    res = rotate(d,center = (100,100,100),angle = .5 )

    
    
