""" scaling images

"""


import logging
logger = logging.getLogger(__name__)

import os
import numpy as np
from gputools import OCLArray, OCLImage, OCLProgram, get_device
from gputools import OCLElementwiseKernel

from _abspath import abspath


from quaternion import Quaternion 



def _mat4_rotation(w=0,x=1,y=0,z=0):
    n = np.array([x,y,z]).astype(np.float32)
    n *= 1./np.sqrt(1.*np.sum(n**2))
    q = Quaternion(np.cos(.5*w),*(np.sin(.5*w)*n))
    return q.toRotation4()


def _mat4_translate(x=0,y=0,z=0):
    M = np.identity(4)
    M[:3,3] = x,y,z
    return M



def affine(data, mat = np.identity(4), mode ="linear"):
    """affine transform data with matrix mat

    """ 

    bop = {"linear":[],"nearest":["-D","USENEAREST"]}

    if not mode in bop.keys():
        raise KeyError("mode = '%s' not defined ,valid: %s"%(mode, bop.keys()))
    
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
    return affine(data,_mat4_translate(x,y,z),mode)


def rotate(data, center = (0,0,0), axis = (1.,0,0), angle = 0., mode ="linear"):
    cz, cy , cx  = center
    m = np.dot(_mat4_translate(cx,cy,cz),
               np.dot(_mat4_rotation(angle,*axis),
                      _mat4_translate(-cx,-cy,-cz)))
    return affine(data, m, mode)



if __name__ == '__main__':
    
    d = np.zeros((200,200,200),np.float32)
    d[20:-20,20:-20,20:-20] = 1.

    # res = translate(d, x = 10, y = 5, z= -10 )
    res = rotate(d,center = (100,100,100),angle = .5 )

    
    
