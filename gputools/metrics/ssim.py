"""

mweigert@mpi-cbg.de
"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from gputools import OCLArray, OCLProgram
from gputools.convolve import uniform_filter
from gputools.utils._abspath import abspath


def compare_ssim_bare(X, Y, data_range=None):
    if not X.dtype == Y.dtype:
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    K1 = 0.01
    K2 = 0.03
    sigma = 1.5

    use_sample_covariance = True

    win_size = 7

    if np.any((np.asarray(X.shape) - win_size) < 0):
        raise ValueError(
            "win_size exceeds image extent.")

    if data_range is None:
        dmin, dmax = np.amin(X), np.amax(X)
        data_range = dmax - dmin

    ndim = X.ndim

    filter_func = uniform_filter
    filter_args = {'size': win_size}

    # ndimage filters need floating point data
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)

    NP = win_size ** ndim

    cov_norm = NP / (NP - 1)  # sample covariance

    # compute (weighted) means
    ux = filter_func(X, **filter_args)
    uy = filter_func(Y, **filter_args)

    # compute (weighted) variances and covariances
    uxx = filter_func(X * X, **filter_args)
    uyy = filter_func(Y * Y, **filter_args)
    uxy = filter_func(X * Y, **filter_args)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    R = data_range
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2

    A1, A2, B1, B2 = ((2 * ux * uy + C1,
                       2 * vxy + C2,
                       ux ** 2 + uy ** 2 + C1,
                       vx + vy + C2))
    D = B1 * B2
    S = (A1 * A2) / D

    # to avoid edge effects will ignore filter radius strip around edges
    pad = (win_size - 1) // 2

    ss = tuple(slice(pad, s - pad) for s in X.shape)
    # compute (weighted) mean of ssim
    mssim = S[ss].mean()

    return mssim


def ssim(x, y, data_range=None):
    """compute ssim
    parameters are like the defaults for skimage.compare_ssim

    """
    if not x.shape == y.shape:
        raise ValueError('Input images must have the same dimensions.')

    K1 = 0.01
    K2 = 0.03
    sigma = 1.5
    win_size = 7

    if np.any((np.asarray(x.shape) - win_size) < 0):
        raise ValueError("win_size exceeds image extent.")

    if data_range is None:
        dmin, dmax = np.amin(x), np.amax(x)
        data_range = dmax - dmin


    x_g = OCLArray.from_array(x.astype(np.float32, copy=False))
    y_g = OCLArray.from_array(y.astype(np.float32, copy=False))

    ndim = x.ndim
    NP = win_size ** ndim
    cov_norm = 1.*NP / (NP - 1)  # sample covariance

    filter_func = uniform_filter
    filter_args = {'size': win_size}

    ux = filter_func(x_g, **filter_args)
    uy = filter_func(y_g, **filter_args)

    # compute (weighted) variances and covariances
    uxx = filter_func(x_g * x_g, **filter_args)
    uyy = filter_func(y_g * y_g, **filter_args)
    uxy = filter_func(x_g * y_g, **filter_args)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    R = 1.*data_range
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2

    A1, A2, B1, B2 = ((2. * ux * uy + C1,
                       2. * vxy + C2,
                       ux ** 2 + uy ** 2 + C1,
                       vx + vy + C2))
    D = B1 * B2
    S = (A1 * A2) / D

    # to avoid edge effects will ignore filter radius strip around edges
    pad = (win_size - 1) // 2

    ss = tuple(slice(pad, s - pad) for s in x.shape)
    # compute (weighted) mean of ssim
    mssim = S.get()[ss].mean()

    return mssim


if __name__ == '__main__':
    pass
