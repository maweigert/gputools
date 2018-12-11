""" scaling images

"""

from __future__ import print_function, unicode_literals, absolute_import, division
import logging

logger = logging.getLogger(__name__)

import os
import numpy as np
import warnings
from gputools import OCLArray, OCLImage, OCLProgram, get_device
from gputools.utils import mat4_rotate, mat4_translate
from ._abspath import abspath


def affine(data, mat=np.identity(4), mode="constant", interpolation="linear"):
    """
    affine transform data with matrix mat, which is the inverse coordinate transform matrix  
    (similar to ndimage.affine_transform)
     
    Parameters
    ----------
    data, ndarray
        3d array to be transformed
    mat, ndarray 
        3x3 or 4x4 inverse coordinate transform matrix 
    mode: string 
        boundary mode, one of the following:
        'constant'
            pads with zeros 
        'edge'
            pads with edge values
        'wrap'
            pads with the repeated version of the input 
    interpolation, string
        interpolation mode, one of the following    
        'linear'
        'nearest'
        
    Returns
    -------
    res: ndarray
        transformed array (same shape as input)
        
    """
    warnings.warn("gputools.transform.affine: API change as of gputools>= 0.2.8: the inverse of the matrix is now used as in scipy.ndimage.affine_transform")

    if not (isinstance(data, np.ndarray) and data.ndim == 3):
        raise ValueError("input data has to be a 3d array!")

    interpolation_defines = {"linear": ["-D", "SAMPLER_FILTER=CLK_FILTER_LINEAR"],
                             "nearest": ["-D", "SAMPLER_FILTER=CLK_FILTER_NEAREST"]}

    mode_defines = {"constant": ["-D", "SAMPLER_ADDRESS=CLK_ADDRESS_CLAMP"],
                    "wrap": ["-D", "SAMPLER_ADDRESS=CLK_ADDRESS_REPEAT"],
                    "edge": ["-D", "SAMPLER_ADDRESS=CLK_ADDRESS_CLAMP_TO_EDGE"]
                    }

    if not interpolation in interpolation_defines:
        raise KeyError(
            "interpolation = '%s' not defined ,valid: %s" % (interpolation, list(interpolation_defines.keys())))

    if not mode in mode_defines:
        raise KeyError("mode = '%s' not defined ,valid: %s" % (mode, list(mode_defines.keys())))

    # reorder matrix, such that x,y,z -> z,y,x (as the kernel is assuming that)

    # inverse of the Matrix
    # mat_inv = np.linalg.inv(mat)


    # # reorder matrix, such that x,y,z -> x,y,z (as the kernel is assuming that)
    # mat_inv_order_xyz = mat_inv[[2, 1, 0, 3]]

    d_im = OCLImage.from_array(data.astype(np.float32, copy = False))
    res_g = OCLArray.empty(data.shape, np.float32)
    mat_inv_g = OCLArray.from_array(mat.astype(np.float32, copy=False))

    prog = OCLProgram(abspath("kernels/transformations.cl")
                      , build_options=interpolation_defines[interpolation] +
                                      mode_defines[mode])

    prog.run_kernel("affine",
                    data.shape[::-1], None,
                    d_im, res_g.data, mat_inv_g.data)

    return res_g.get()


def shift(data, shift=(0, 0, 0), mode="constant", interpolation="linear"):
    """
    translates 3d data by given amount
  
    
    Parameters
    ----------
    data: ndarray
        3d array
    shift : float or sequence
        The shift along the axes. If a float, `shift` is the same for each axis. 
        If a sequence, `shift` should contain one value for each axis.    
    mode: string 
        boundary mode, one of the following:      
        'constant'
            pads with zeros 
        'edge'
            pads with edge values
        'wrap'
            pads with the repeated version of the input 
    interpolation, string
        interpolation mode, one of the following       
        'linear'
        'nearest'
        
    Returns
    -------
    res: ndarray
        shifted array (same shape as input)
    """
    if np.isscalar(shift):
        shift = (shift,)*3

    if len(shift) != 3:
        raise ValueError("shift (%s) should be of length 3!")

    shift = -np.array(shift)
    return affine(data, mat4_translate(*shift), mode=mode, interpolation=interpolation)


def rotate(data, axis=(1., 0, 0), angle=0., center=None, mode="constant", interpolation="linear"):
    """
    rotates data around axis by a given angle

    Parameters
    ----------
    data: ndarray
        3d array
    axis: tuple
        axis to rotate by angle about
        axis = (x,y,z)
    angle: float
    center: tuple or None
        origin of rotation (cz,cy,cx) in pixels
        if None, center is the middle of data
    
    mode: string 
        boundary mode, one of the following:        
        'constant'
            pads with zeros 
        'edge'
            pads with edge values
        'wrap'
            pads with the repeated version of the input 
    interpolation, string
        interpolation mode, one of the following      
        'linear'
        'nearest'
        
    Returns
    -------
    res: ndarray
        rotated array (same shape as input)

    """
    if center is None:
        center = tuple([s // 2 for s in data.shape])

    cx, cy, cz = center
    m = np.dot(mat4_translate(cx, cy, cz),
               np.dot(mat4_rotate(angle, *axis),
                      mat4_translate(-cx, -cy, -cz)))
    m = np.linalg.inv(m)
    return affine(data, m, mode=mode, interpolation=interpolation)


if __name__ == '__main__':
    d = np.zeros((200, 200, 200), np.float32)
    d[20:-20, 20:-20, 20:-20] = 1.

    # res = translate(d, x = 10, y = 5, z= -10 )
    res = rotate(d, center=(100, 100, 100), angle=.5)
