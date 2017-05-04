from __future__ import print_function, unicode_literals, absolute_import, division
import logging

logger = logging.getLogger(__name__)

import os
import numpy as np
from mako.template import Template

from gputools import OCLArray, OCLProgram, get_device

from gputools.core.ocltypes import assert_bufs_type
from gputools.utils.tile_iterator import tile_iterator
from ._abspath import abspath


def _filter_generic_2_gpu(FUNC = "fmax(res,val)", DEFAULT = "-INFINITY"):
    def _filt(data_g, size=(3, 3), res_g=None):
        assert_bufs_type(np.float32, data_g)

        with open(abspath("kernels/generic_reduce_filter.cl"),"r") as f:
            tpl = Template(f.read())

        rendered = tpl.render(FSIZE_X = size[-1],FSIZE_Y = size[-2],FSIZE_Z = 1,
                              FUNC = FUNC, DEFAULT = DEFAULT)

        prog = OCLProgram(src_str=rendered)

        tmp_g = OCLArray.empty_like(data_g)

        if res_g is None:
            res_g = OCLArray.empty_like(data_g)

        prog.run_kernel("filter_2_x", data_g.shape[::-1], None, data_g.data, tmp_g.data)
        prog.run_kernel("filter_2_y", data_g.shape[::-1], None, tmp_g.data, res_g.data)

        return res_g
    return _filt

def _filter_max_2_gpu(data_g, size=(3, 3), res_g=None):
    _filt = _filter_generic_2_gpu(FUNC = "fmax(res,val)", DEFAULT = "-INFINITY")
    return _filt(data_g, size=(3, 3), res_g=None)


def _filter_mean_2_gpu(data_g, size=(3, 3), res_g=None):
    _filt = _filter_generic_2_gpu(FUNC = "res +val", DEFAULT = "0.f")
    return _filt(data_g, size=(3, 3), res_g=None)




def _generic_filter_gpu_2d(FUNC = "fmax(res,val)", DEFAULT = "-INFINITY"):
    def _filt(data_g, size=(3, 3), res_g=None):
        assert_bufs_type(np.float32, data_g)

        with open(abspath("kernels/generic_reduce_filter.cl"), "r") as f:
            tpl = Template(f.read())

        rendered = tpl.render(FSIZE_X=size[-1], FSIZE_Y=size[-2], FSIZE_Z=1,
                              FUNC=FUNC, DEFAULT=DEFAULT)

        prog = OCLProgram(src_str=rendered)

        tmp_g = OCLArray.empty_like(data_g)

        if res_g is None:
            res_g = OCLArray.empty_like(data_g)

        prog.run_kernel("filter_2_x", data_g.shape[::-1], None, data_g.data, tmp_g.data)
        prog.run_kernel("filter_2_y", data_g.shape[::-1], None, tmp_g.data, res_g.data)
        return res_g
    return _filt

def _generic_filter_gpu_3d(FUNC = "fmax(res,val)", DEFAULT = "-INFINITY"):
    def _filt(data_g, size=(3, 3,3 ), res_g=None):
        assert_bufs_type(np.float32, data_g)

        with open(abspath("kernels/generic_reduce_filter.cl"), "r") as f:
            tpl = Template(f.read())

        rendered = tpl.render(FSIZE_X=size[-1], FSIZE_Y=size[-2], FSIZE_Z=size[-3],
                              FUNC=FUNC, DEFAULT=DEFAULT)

        prog = OCLProgram(src_str=rendered)

        tmp_g = OCLArray.empty_like(data_g)

        if res_g is None:
            res_g = OCLArray.empty_like(data_g)

        prog.run_kernel("filter_3_x", data_g.shape[::-1], None, data_g.data, res_g.data)
        prog.run_kernel("filter_3_y", data_g.shape[::-1], None, res_g.data, tmp_g.data)
        prog.run_kernel("filter_3_z", data_g.shape[::-1], None, tmp_g.data, res_g.data)
        return res_g
    return _filt



def make_filter(filter_gpu):
    def _filter_numpy(data, size):
        data_g = OCLArray.from_array(data.astype(np.float32))
        return filter_gpu(data_g = data_g, size=size).get()

    def _filter(data, size=10, res_g=None, sub_blocks=(1, 1, 1)):
        if np.isscalar(size):
            size = (size,)*len(data.shape)

        if isinstance(data, np.ndarray):
            if set(sub_blocks) == {1} or sub_blocks is None:
                return _filter_numpy(data, size)
            else:
                # cut the image into tile and operate on every of them
                N_sub = [int(np.ceil(1. * n / s)) for n, s in zip(data.shape, sub_blocks)]
                Npads = int(size // 2)
                res = np.empty(data.shape, np.float32)
                for i, (data_tile, data_s_src, data_s_dest) \
                        in enumerate(tile_iterator(data, blocksize=N_sub,
                                                   padsize=Npads,
                                                   mode="constant")):
                    res_tile = _filter_numpy(data_tile.copy(),
                                                 size)
                    res[data_s_src] = res_tile[data_s_dest]
                return res

        elif isinstance(data, OCLArray):
            return filter_gpu(data, size=size, res_g=res_g)
        else:
            raise TypeError("array argument (1) has bad type: %s" % type(data))

    return _filter


####################################################################################



def max_filter(data, size=7, res_g=None, sub_blocks=(1, 1, 1)):
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
    if data.ndim == 2:
        _filt = make_filter(_generic_filter_gpu_2d(FUNC = "fmax(res,val)", DEFAULT = "-INFINITY"))
    elif data.ndim == 3:
        _filt = make_filter(_generic_filter_gpu_3d(FUNC = "fmax(res,val)", DEFAULT = "-INFINITY"))

    return _filt(data = data, size = size, res_g = res_g, sub_blocks=sub_blocks)


def min_filter(data, size=7, res_g=None, sub_blocks=(1, 1, 1)):
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
    if data.ndim == 2:
        _filt = make_filter(_generic_filter_gpu_2d(FUNC="fmin(res,val)", DEFAULT="INFINITY"))
    elif data.ndim == 3:
        _filt = make_filter(_generic_filter_gpu_3d(FUNC="fmin(res,val)", DEFAULT="INFINITY"))

    return _filt(data=data, size=size, res_g=res_g, sub_blocks=sub_blocks)



def uniform_filter(data, size=7, res_g=None, sub_blocks=(1, 1, 1), normalized = True):
    """
        mean filter of given size

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
    normalized: bool
        if True, the filter corresponds to mean
        if False, the filter corresponds to sum

    Returns
    -------
        filtered image or None (if OCLArray)
    """



    if data.ndim == 2:
        _filt = make_filter(_generic_filter_gpu_2d(FUNC="res+val" , DEFAULT="0"))
    elif data.ndim == 3:
        _filt = make_filter(_generic_filter_gpu_3d(FUNC="res+val", DEFAULT="0"))

    res =  _filt(data=data, size=size, res_g=res_g, sub_blocks=sub_blocks)

    if normalized:
        if np.isscalar(size):
            normed = size**len(data.shape)
        else:
            normed = np.prod(size)

        res /= 1.*normed

    return res


        #
# def _filter_max_3_gpu(data_g, size=10, res_g=None):
#     assert_bufs_type(np.float32, data_g)
#
#     prog = OCLProgram(abspath("kernels/minmax_filter.cl"))
#
#     tmp_g = OCLArray.empty_like(data_g)
#
#     if res_g is None:
#         res_g = OCLArray.empty_like(data_g)
#
#     prog.run_kernel("max_3_x", data_g.shape[::-1], None, data_g.data, res_g.data, np.int32(size[-1]))
#     prog.run_kernel("max_3_y", data_g.shape[::-1], None, res_g.data, tmp_g.data, np.int32(size[-2]))
#     prog.run_kernel("max_3_z", data_g.shape[::-1], None, tmp_g.data, res_g.data, np.int32(size[-3]))
#
#     return res_g
#
#
# def _max_filter_gpu(data_g, size=5, res_g=None):
#     assert_bufs_type(np.float32, data_g)
#
#     assert (len(data_g.shape) == len(size))
#
#     if len(data_g.shape) == 2:
#         return _filter_max_2_gpu(data_g, size=size, res_g=res_g)
#     elif len(data_g.shape) == 3:
#         return _filter_max_3_gpu(data_g, size=size, res_g=res_g)
#     else:
#         raise NotImplementedError("only 2 or 3d arrays are supported for now")
#
#
# def _max_filter_numpy(data, size=5):
#     data_g = OCLArray.from_array(data.astype(np.float32))
#     return _max_filter_gpu(data_g, size=size).get()
#
#
# def max_filter(data, size=10, res_g=None, sub_blocks=(1, 1, 1)):
#     """
#         maximum filter of given size
#
#     Parameters
#     ----------
#     data: 2 or 3 dimensional ndarray or OCLArray of type float32
#         input data
#     size: scalar, tuple
#         the size of the patch to consider
#     res_g: OCLArray
#         store result in buffer if given
#     sub_blocks:
#         perform over subblock tiling (only if data is ndarray)
#
#     Returns
#     -------
#         filtered image or None (if OCLArray)
#     """
#
#     if np.isscalar(size):
#         size = (size,) * len(data.shape)
#
#     if isinstance(data, np.ndarray):
#         if set(sub_blocks) == {1} or sub_blocks is None:
#             return _max_filter_numpy(data, size)
#         else:
#             # cut the image into tile and operate on every of them
#             N_sub = [int(np.ceil(1. * n / s)) for n, s in zip(data.shape, sub_blocks)]
#             Npads = int(size // 2)
#             res = np.empty(data.shape, np.float32)
#             for i, (data_tile, data_s_src, data_s_dest) \
#                     in enumerate(tile_iterator(data, blocksize=N_sub,
#                                                padsize=Npads,
#                                                mode="constant")):
#                 res_tile = _max_filter_numpy(data_tile.copy(),
#                                              size)
#                 res[data_s_src] = res_tile[data_s_dest]
#             return res
#
#
#     elif isinstance(data, OCLArray):
#         return _max_filter_gpu(data, size=size, res_g=res_g)
#     else:
#         raise TypeError("array argument (1) has bad type: %s" % type(data))
#
#
# def _filter_min_2_gpu(data_g, size=(10, 10), res_g=None):
#     assert_bufs_type(np.float32, data_g)
#
#     prog = OCLProgram(abspath("kernels/minmax_filter.cl"))
#
#     tmp_g = OCLArray.empty_like(data_g)
#
#     if res_g is None:
#         res_g = OCLArray.empty_like(data_g)
#
#     prog.run_kernel("min_2_x", data_g.shape[::-1], None, data_g.data, tmp_g.data, np.int32(size[-1]))
#     prog.run_kernel("min_2_y", data_g.shape[::-1], None, tmp_g.data, res_g.data, np.int32(size[-2]))
#
#     return res_g
#
#
# def _filter_min_3_gpu(data_g, size=(10, 10, 10), res_g=None):
#     assert_bufs_type(np.float32, data_g)
#
#     prog = OCLProgram(abspath("kernels/minmax_filter.cl"))
#
#     tmp_g = OCLArray.empty_like(data_g)
#
#     if res_g is None:
#         res_g = OCLArray.empty_like(data_g)
#
#     prog.run_kernel("min_3_x", data_g.shape[::-1], None, data_g.data, res_g.data, np.int32(size[-1]))
#     prog.run_kernel("min_3_y", data_g.shape[::-1], None, res_g.data, tmp_g.data, np.int32(size[-2]))
#     prog.run_kernel("min_3_z", data_g.shape[::-1], None, tmp_g.data, res_g.data, np.int32(size[-3]))
#
#     return res_g
#
#
# def _min_filter_gpu(data_g, size=(10, 10), res_g=None):
#     assert_bufs_type(np.float32, data_g)
#
#     assert (len(data_g.shape) == len(size))
#
#     if len(data_g.shape) == 2:
#         return _filter_min_2_gpu(data_g, size=size, res_g=res_g)
#     elif len(data_g.shape) == 3:
#         return _filter_min_3_gpu(data_g, size=size, res_g=res_g)
#     else:
#         raise NotImplementedError("only 2 or 3d arrays are supported for now")
#
#
# def _min_filter_numpy(data, size=(10, 10)):
#     data_g = OCLArray.from_array(data.astype(np.float32))
#     return _min_filter_gpu(data_g, size=size).get()
#
#
# def min_filter(data, size=10, res_g=None, sub_blocks=(1, 1, 1)):
#     """
#         minimum filter of given size
#
#     Parameters
#     ----------
#     data: 2 or 3 dimensional ndarray or OCLArray of type float32
#         input data
#     size: scalar, tuple
#         the size of the patch to consider
#     res_g: OCLArray
#         store result in buffer if given
#     sub_blocks:
#         perform over subblock tiling (only if data is ndarray)
#
#     Returns
#     -------
#         filtered image or None (if OCLArray)
#     """
#
#     if np.isscalar(size):
#         size = (size,) * len(data.shape)
#
#     if isinstance(data, np.ndarray):
#         if set(sub_blocks) == {1} or sub_blocks is None:
#             return _min_filter_numpy(data, size)
#         else:
#             # cut the image into tile and operate on every of them
#             N_sub = [int(np.ceil(1. * n / s)) for n, s in zip(data.shape, sub_blocks)]
#             Npads = int(size // 2)
#             res = np.empty(data.shape, np.float32)
#             for i, (data_tile, data_s_src, data_s_dest) \
#                     in enumerate(tile_iterator(data, blocksize=N_sub,
#                                                padsize=Npads,
#                                                mode="constant")):
#                 res_tile = _min_filter_numpy(data_tile.copy(),
#                                              size)
#                 res[data_s_src] = res_tile[data_s_dest]
#             return res
#
#
#     elif isinstance(data, OCLArray):
#         return _min_filter_gpu(data, size=size, res_g=res_g)
#     else:
#         raise TypeError("array argument (1) has bad type: %s" % type(data))



if __name__ == '__main__':

    x = np.random.uniform(-1,1,(100,100))
    x_g = OCLArray.from_array(x.astype(np.float32))
    _filter_mean_2_gpu(x_g)