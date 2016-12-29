from __future__ import print_function, unicode_literals, absolute_import, division
import logging

logger = logging.getLogger(__name__)

import os
import numpy as np
from gputools import OCLArray, OCLProgram, get_device

from gputools.core.ocltypes import assert_bufs_type
from gputools.utils.tile_iterator import tile_iterator
from ._abspath import abspath


def _filter_max_2_gpu(data_g, size=10, res_g=None):
    assert_bufs_type(np.float32, data_g)

    prog = OCLProgram(abspath("kernels/minmax_filter.cl"))

    tmp_g = OCLArray.empty_like(data_g)

    if res_g is None:
        res_g = OCLArray.empty_like(data_g)

    prog.run_kernel("max_2_x", data_g.shape[::-1], None, data_g.data, tmp_g.data, np.int32(size[-1]))
    prog.run_kernel("max_2_y", data_g.shape[::-1], None, tmp_g.data, res_g.data, np.int32(size[-2]))

    return res_g


def _filter_max_3_gpu(data_g, size=10, res_g=None):
    assert_bufs_type(np.float32, data_g)

    prog = OCLProgram(abspath("kernels/minmax_filter.cl"))

    tmp_g = OCLArray.empty_like(data_g)

    if res_g is None:
        res_g = OCLArray.empty_like(data_g)

    prog.run_kernel("max_3_x", data_g.shape[::-1], None, data_g.data, res_g.data, np.int32(size[-1]))
    prog.run_kernel("max_3_y", data_g.shape[::-1], None, res_g.data, tmp_g.data, np.int32(size[-2]))
    prog.run_kernel("max_3_z", data_g.shape[::-1], None, tmp_g.data, res_g.data, np.int32(size[-3]))

    return res_g




def _max_filter_gpu(data_g, size=5, res_g=None):
    assert_bufs_type(np.float32, data_g)

    assert (len(data_g.shape) == len(size))

    if len(data_g.shape) == 2:
        return _filter_max_2_gpu(data_g, size=size, res_g=res_g)
    elif len(data_g.shape) == 3:
        return _filter_max_3_gpu(data_g, size=size, res_g=res_g)
    else:
         raise NotImplementedError("only 2 or 3d arrays are supported for now")


def _max_filter_numpy(data, size=5):
    data_g = OCLArray.from_array(data.astype(np.float32))
    return _max_filter_gpu(data_g, size=size).get()


def max_filter(data, size=10, res_g=None, sub_blocks=(1, 1, 1)):
    """
        maximum filter of given size

    Parameters
    ----------
    data: 2 or 3 dimensional ndarray or OCLArray of type float32
        input data
    size: scalar, tuple
        the size of the patch to consider
    res_g: OCLArray
        store result in buffer if given
    sub_blocks:
        perform over subblock tiling (only if data is ndarray)

    Returns
    -------
        filtered image or None (if OCLArray)
    """

    if np.isscalar(size):
        size = (size,)*len(data.shape)

    if isinstance(data, np.ndarray):
        if set(sub_blocks) == {1} or sub_blocks is None:
            return _max_filter_numpy(data, size)
        else:
            # cut the image into tile and operate on every of them
            N_sub = [int(np.ceil(1. * n / s)) for n, s in zip(data.shape, sub_blocks)]
            Npads = int(size // 2)
            res = np.empty(data.shape, np.float32)
            for i, (data_tile, data_s_src, data_s_dest) \
                    in enumerate(tile_iterator(data, blocksize=N_sub,
                                               padsize=Npads,
                                               mode="constant")):
                res_tile = _max_filter_numpy(data_tile.copy(),
                                             size)
                res[data_s_src] = res_tile[data_s_dest]
            return res


    elif isinstance(data, OCLArray):
        return _max_filter_gpu(data, size=size, res_g=res_g)
    else:
        raise TypeError("array argument (1) has bad type: %s" % type(data))



def _filter_min_2_gpu(data_g, size=(10,10), res_g=None):
    assert_bufs_type(np.float32, data_g)

    prog = OCLProgram(abspath("kernels/minmax_filter.cl"))

    tmp_g = OCLArray.empty_like(data_g)

    if res_g is None:
        res_g = OCLArray.empty_like(data_g)

    prog.run_kernel("min_2_x", data_g.shape[::-1], None, data_g.data, tmp_g.data, np.int32(size[-1]))
    prog.run_kernel("min_2_y", data_g.shape[::-1], None, tmp_g.data, res_g.data, np.int32(size[-2]))

    return res_g


def _filter_min_3_gpu(data_g, size=(10,10,10), res_g=None):
    assert_bufs_type(np.float32, data_g)

    prog = OCLProgram(abspath("kernels/minmax_filter.cl"))

    tmp_g = OCLArray.empty_like(data_g)

    if res_g is None:
        res_g = OCLArray.empty_like(data_g)

    prog.run_kernel("min_3_x", data_g.shape[::-1], None, data_g.data, res_g.data, np.int32(size[-1]))
    prog.run_kernel("min_3_y", data_g.shape[::-1], None, res_g.data, tmp_g.data, np.int32(size[-2]))
    prog.run_kernel("min_3_z", data_g.shape[::-1], None, tmp_g.data, res_g.data, np.int32(size[-3]))

    return res_g




def _min_filter_gpu(data_g, size=(10,10), res_g=None):
    assert_bufs_type(np.float32, data_g)

    assert (len(data_g.shape)==len(size))

    if len(data_g.shape) == 2:
        return _filter_min_2_gpu(data_g, size=size, res_g=res_g)
    elif len(data_g.shape) == 3:
        return _filter_min_3_gpu(data_g, size=size, res_g=res_g)
    else:
         raise NotImplementedError("only 2 or 3d arrays are supported for now")


def _min_filter_numpy(data, size=(10,10)):
    data_g = OCLArray.from_array(data.astype(np.float32))
    return _min_filter_gpu(data_g, size=size).get()


def min_filter(data, size=10, res_g=None, sub_blocks=(1, 1, 1)):
    """
        minimum filter of given size

    Parameters
    ----------
    data: 2 or 3 dimensional ndarray or OCLArray of type float32
        input data
    size: scalar, tuple
        the size of the patch to consider
    res_g: OCLArray
        store result in buffer if given
    sub_blocks:
        perform over subblock tiling (only if data is ndarray)

    Returns
    -------
        filtered image or None (if OCLArray)
    """

    if np.isscalar(size):
        size = (size,)*len(data.shape)

    if isinstance(data, np.ndarray):
        if set(sub_blocks) == {1} or sub_blocks is None:
            return _min_filter_numpy(data, size)
        else:
            # cut the image into tile and operate on every of them
            N_sub = [int(np.ceil(1. * n / s)) for n, s in zip(data.shape, sub_blocks)]
            Npads = int(size // 2)
            res = np.empty(data.shape, np.float32)
            for i, (data_tile, data_s_src, data_s_dest) \
                    in enumerate(tile_iterator(data, blocksize=N_sub,
                                               padsize=Npads,
                                               mode="constant")):
                res_tile = _min_filter_numpy(data_tile.copy(),
                                             size)
                res[data_s_src] = res_tile[data_s_dest]
            return res


    elif isinstance(data, OCLArray):
        return _min_filter_gpu(data, size=size, res_g=res_g)
    else:
        raise TypeError("array argument (1) has bad type: %s" % type(data))
