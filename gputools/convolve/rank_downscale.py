from __future__ import print_function, unicode_literals, absolute_import, division
import logging
from typing import Union, Tuple

logger = logging.getLogger(__name__)

import numpy as np
from mako.template import Template
from numbers import Number
from gputools import OCLArray, OCLProgram
from gputools.core.ocltypes import cl_buffer_datatype_dict
from ._abspath import abspath


def _rank_downscale_2d(data_g, size=(3, 3), rank=None, cval = 0, res_g=None):
    if not data_g.dtype.type in cl_buffer_datatype_dict:
        raise ValueError("dtype %s not supported" % data_g.dtype.type)

    DTYPE = cl_buffer_datatype_dict[data_g.dtype.type]

    size = tuple(map(int, size)) 
    
    if rank is None:
        rank = np.prod(size) // 2
    
    with open(abspath("kernels/rank_downscale.cl"), "r") as f:
        tpl = Template(f.read())

    rendered = tpl.render(DTYPE = DTYPE,FSIZE_Z=0, FSIZE_X=size[1], FSIZE_Y=size[0],CVAL = cval)

    prog = OCLProgram(src_str=rendered)

    if res_g is None:
        res_g = OCLArray.empty(tuple(s0//s for s, s0 in zip(size,data_g.shape)), data_g.dtype)

    prog.run_kernel("rank_2", res_g.shape[::-1], None, data_g.data, res_g.data, 
                    np.int32(data_g.shape[1]), np.int32(data_g.shape[0]),
                    np.int32(rank))
    return res_g

def _rank_downscale_3d(data_g, size=(3, 3, 3), rank=None, cval = 0, res_g=None):
    if not data_g.dtype.type in cl_buffer_datatype_dict:
        raise ValueError("dtype %s not supported" % data_g.dtype.type)

    DTYPE = cl_buffer_datatype_dict[data_g.dtype.type]

    size = tuple(map(int, size)) 
    
    if rank is None:
        rank = np.prod(size) // 2
    
    with open(abspath("kernels/rank_downscale.cl"), "r") as f:
        tpl = Template(f.read())

    rendered = tpl.render(DTYPE = DTYPE,FSIZE_X=size[2], FSIZE_Y=size[1], FSIZE_Z=size[0],CVAL = cval)

    prog = OCLProgram(src_str=rendered)

    if res_g is None:
        res_g = OCLArray.empty(tuple(s0//s for s, s0 in zip(size,data_g.shape)), data_g.dtype)

    prog.run_kernel("rank_3", res_g.shape[::-1], None, data_g.data, res_g.data, 
                    np.int32(data_g.shape[2]), np.int32(data_g.shape[1]), np.int32(data_g.shape[0]),
                    np.int32(rank))
    return res_g


def rank_downscale(data:np.ndarray, size:Union[int, Tuple[int]]=3, rank=None):
    """
        downscales an image by the given factor and returns the rank-th element in each block

    Parameters
    ----------
    data: numpy.ndarray
        input data (2d or 3d)
    size: int or tuple
        downsampling factors
    rank: int
        rank of element to retain 
            rank = 0     -> minimum 
            rank = -1    -> maximum 
            rank = None  -> median 
        
    Returns
    -------
        downscaled image 
    """

    if not (isinstance(data, np.ndarray) and data.ndim in (2,3)):
        raise ValueError("input data has to be a 2d or 3d numpy array!")

    if isinstance(size, Number):
        size = (int(size),)*data.ndim
        
    if len(size) != data.ndim:
        raise ValueError("size has to be a tuple of 3 ints")
    
    if rank is None:
        rank = np.prod(size) // 2
    else:
        rank = rank % np.prod(size)

    data_g = OCLArray.from_array(data)
    
    if data.ndim==2:
        res_g = _rank_downscale_2d(data_g, size=size, rank=rank)
    elif data.ndim==3:
        res_g = _rank_downscale_3d(data_g, size=size, rank=rank)
    else: 
        raise ValueError("data has to be 2d or 3d")
    
    return res_g.get()


