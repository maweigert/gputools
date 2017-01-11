from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import numpy.testing as npt
import gputools
import scipy.ndimage.filters as sp_filter


def test_conv_gpu():
    N = 128
    d = np.zeros((N, N + 3, N + 5), np.float32)

    d[N // 2, N // 2, N // 2] = 1.

    h = np.exp(-10 * np.linspace(-1, 1, 17) ** 2)

    res = gputools.convolve_sep3(d, h, h, h)


def test_conv_sep2_numpy():
    Nx, Ny = 128, 200

    d = np.zeros((Ny, Nx), np.float32)

    d[::10, ::10] = 1.

    hx = np.ones(8)
    hy = np.ones(3)

    res1 = gputools.convolve_sep2(d, hx, hy, sub_blocks=(1, 1))
    res2 = gputools.convolve_sep2(d, hx, hy, sub_blocks=(2, 11))

    assert np.allclose(res1, res2)
    return res1, res2


def test_conv_sep3_numpy():
    Nz, Nx, Ny = 128, 203, 303

    d = np.zeros((Nz, Ny, Nx), np.float32)

    d[::10, ::10, ::10] = 1.

    hx = np.ones(8)
    hy = np.ones(3)
    hz = np.ones(11)

    res1 = gputools.convolve_sep3(d, hx, hy, hz, sub_blocks=(1, 1, 1))
    res2 = gputools.convolve_sep3(d, hx, hy, hz, sub_blocks=(7, 4, 3))

    assert np.allclose(res1, res2)
    return res1, res2


def _conv_sep2_numpy(d, hx, hy):
    tmp = sp_filter.convolve(d, hx.reshape((1, len(hx))), mode="constant")
    return sp_filter.convolve(tmp, hy.reshape((len(hy), 1)), mode="constant")


def _conv_sep3_numpy(d, hx, hy, hz):
    res = sp_filter.convolve(d, hx.reshape((1, 1, len(hx))), mode="constant")
    tmp = sp_filter.convolve(res, hy.reshape((1, len(hy), 1)), mode="constant")
    return sp_filter.convolve(tmp, hz.reshape((len(hz), 1, 1)), mode="constant")


def _convolve_rand(dshape, hshapes):
    print("convolving test: dshape = %s, hshapes  = %s" % (dshape, hshapes))
    np.random.seed(1)
    d = np.random.uniform(0, 1, dshape).astype(np.float32)

    hs = [np.random.uniform(1, 2, hshape).astype(np.float32) for hshape in hshapes]

    if len(hs) == 2:
        out1 = _conv_sep2_numpy(d, *hs)
        out2 = gputools.convolve_sep2(d, *hs)
    elif len(hs) == 3:
        out1 = _conv_sep3_numpy(d, *hs)
        out2 = gputools.convolve_sep3(d, *hs)

    print("difference: ", np.amax(np.abs(out1-out2)))

    npt.assert_allclose(out1, out2, rtol=1.e-2, atol=1.e-5)

    return out1, out2


def test_simple2():
    d = np.zeros((100, 100))
    d[50, 50] = 1.
    hx = np.random.uniform(0, 1, 11)
    hy = np.random.uniform(0, 1, 11)
    res1 = _conv_sep2_numpy(d, hx, hy)
    res2 = gputools.convolve_sep2(d, hx, hy)
    return res1, res2


def test_simple3():
    d = np.zeros((100, 100,100))
    d[50, 50, 50] = 1.
    hx = np.random.uniform(0, 1, 7)
    hy = np.random.uniform(0, 1, 11)
    hz = np.random.uniform(0, 1, 17)

    # hy = np.ones(1)
    # hz = np.ones(1)

    res1 = _conv_sep3_numpy(d, hx, hy, hz)
    res2 = gputools.convolve_sep3(d, hx, hy, hz)
    return res1, res2


def test_all():
    for ndim in [2, 3]:
        for N in range(30, 200, 40):
            for Nh in range(3, 11, 2):
                dshape = [N // ndim + 3 * n for n in range(ndim)]
                hshape = [Nh + 3 * n for n in range(ndim)]

                _convolve_rand(dshape, hshape)


if __name__ == '__main__':
    # res1, res2 = test_conv_sep2_numpy()
    # res1, res2 = test_conv_sep3_numpy()

    # res1, res2 = _convolve_rand((100,200),(11,22))

    res1, res2 = test_simple3()

