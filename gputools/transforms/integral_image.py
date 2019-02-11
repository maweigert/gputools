"""

mweigert@mpi-cbg.de
"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import math
import pyopencl as cl
from gputools import OCLProgram, OCLArray, get_device
from gputools.utils import next_power_of_2
from ._abspath import abspath

def integral_image(x):
    if x.ndim == 2:
        return _integral2(x)
    elif x.ndim == 3:
        return _integral3(x)


def _integral2(x):
    assert x.dtype.type == np.float32

    dtype_itemsize = x.dtype.itemsize

    max_local_size = get_device().get_info("MAX_WORK_GROUP_SIZE")

    prog = OCLProgram(abspath("kernels/integral_image.cl"))
    ny, nx = x.shape
    x_g = OCLArray.from_array(x)
    tmp_g = OCLArray.empty_like(x)
    y_g = OCLArray.empty_like(x)

    def _scan_single(src, dst, ns, strides):
        nx, ny = ns
        stride_x, stride_y = strides
        loc = min(next_power_of_2(nx // 2), max_local_size // 2)
        nx_block = 2 * loc
        nx_pad = math.ceil(nx / nx_block) * nx_block

        nblocks = math.ceil(nx_pad // 2 / loc)
        sum_blocks = OCLArray.empty((ny, nblocks), dst.dtype)
        shared = cl.LocalMemory(2 * dtype_itemsize * loc)
        for b in range(nblocks):
            offset = b * loc
            prog.run_kernel("scan2d", (loc, ny), (loc, 1),
                            src.data, dst.data, sum_blocks.data, shared,
                            np.int32(nx_block), np.int32(stride_x), np.int32(stride_y), np.int32(offset), np.int32(b),
                            np.int32(nblocks), np.int32(nx))
        if nblocks > 1:
            _scan_single(sum_blocks, sum_blocks, (nblocks, ny), (1, nblocks))
            prog.run_kernel("add_sums2d", (nx_pad, ny), (nx_block, 1),
                            sum_blocks.data, dst.data,
                            np.int32(stride_x), np.int32(stride_y), np.int32(nblocks), np.int32(nx))

    _scan_single(x_g, tmp_g, (nx, ny), (1, nx))
    _scan_single(tmp_g, y_g, (ny, nx), (nx, 1))

    return y_g.get()


def _integral3(x):
    assert x.dtype.type == np.float32

    dtype_itemsize = x.dtype.itemsize

    max_local_size = get_device().get_info("MAX_WORK_GROUP_SIZE")

    prog = OCLProgram(abspath("kernels/integral_image.cl"))
    nz, ny, nx = x.shape
    x_g = OCLArray.from_array(x)
    tmp_g = OCLArray.empty_like(x)
    y_g = OCLArray.empty_like(x)

    def _scan_single(src, dst, ns, strides):
        nx, ny, nz = ns
        stride_x, stride_y, stride_z = strides
        loc = min(next_power_of_2(nx // 2), max_local_size // 2)
        nx_block = 2 * loc
        nx_pad = math.ceil(nx / nx_block) * nx_block

        nblocks = math.ceil(nx_pad // 2 / loc)
        sum_blocks = OCLArray.empty((nz, ny, nblocks), dst.dtype)
        shared = cl.LocalMemory(2 * dtype_itemsize * loc)
        for b in range(nblocks):
            offset = b * loc
            prog.run_kernel("scan3d", (loc, ny, nz), (loc, 1, 1),
                            src.data, dst.data, sum_blocks.data, shared,
                            np.int32(nx_block),
                            np.int32(stride_x), np.int32(stride_y), np.int32(stride_z), np.int32(offset), np.int32(b),
                            np.int32(nblocks), np.int32(ny), np.int32(nx))
        if nblocks > 1:
            _scan_single(sum_blocks, sum_blocks, (nblocks, ny, nz), (1, nblocks, nblocks * ny))
            prog.run_kernel("add_sums3d", (nx_pad, ny, nz), (nx_block, 1, 1),
                            sum_blocks.data, dst.data,
                            np.int32(stride_x), np.int32(stride_y), np.int32(stride_z),
                            np.int32(nblocks), np.int32(ny), np.int32(nx))

    _scan_single(x_g, y_g, (nx, ny, nz), (1, nx, nx * ny))
    _scan_single(y_g, tmp_g, (ny, nx, nz), (nx, 1, nx * ny))
    _scan_single(tmp_g, y_g, (nz, nx, ny), (ny * nx, 1, nx))

    return y_g.get()
