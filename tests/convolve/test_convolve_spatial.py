"""


mweigert@mpi-cbg.de

"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from time import time
from functools import reduce
from gputools import convolve_spatial2, convolve_spatial3


def create_psf(sig=(.1,.1), xy_angle = 0., N = 10):
    x = np.linspace(-1,1,N+1)[:-1]
    Y, X = np.meshgrid(x,x,indexing="ij")
    X2 = np.cos(xy_angle)*X - np.sin(xy_angle)*Y
    Y2 = np.cos(xy_angle)*Y + np.sin(xy_angle)*X
    h = np.exp(-reduce(np.add,[_X**2/_s**2/2. for _X,_s in zip([Y2,X2],sig)]))
    h *= 1./np.sum(h)
    return h


def psf_grid_motion(Gx,Gy,N = 20):
    return np.stack([np.stack([create_psf(sig = (.01+.1*np.sqrt(_x**2+_y**2),
                                          .01+.4*np.sqrt(_x**2+_y**2)),
                                          N = N,
                                          xy_angle = -1.*np.pi+np.arctan2(_y,_x))\
                             for _y in np.linspace(-1,1,Gy)]) for _x in np.linspace(-1,1,Gx)])


def psf_grid_const(Gx,Gy,N=21, sx = 0.01, sy = 0.01):
    return np.stack([np.stack([create_psf(w = 0,N = N,
                                        sx = sx, sy = sy)
                             for _y in np.linspace(-1,1,Gy)])  for _x in np.linspace(-1,1,Gx)])


def create_psf3(sig = (.3,.3,.3), N = 10, xy_angle = 0.):
    x = np.linspace(-1,1,N+1)[:-1]
    Z,Y,X = np.meshgrid(x,x,x,indexing="ij")
    X2 = np.cos(xy_angle)*X - np.sin(xy_angle)*Y
    Y2 = np.cos(xy_angle)*Y + np.sin(xy_angle)*X
    h = np.exp(-reduce(np.add,[_X**2/_s**2/2. for _X,_s in zip([Z,Y2,X2],sig)]))
    h *= 1./np.sum(h)
    return h


def make_grid2(hs):
    Gy,Gx, Hy, Hx = hs.shape

    im = np.zeros((Gy*Hy,Gx*Hx))
    for i in range(Gx):
        for j in range(Gy):
            im[j*Hy:(j+1)*Hy,i*Hx:(i+1)*Hx] = hs[j,i]

    return im


def psf_grid_const3(Gx,Gy,Gz, N=21, sig = (0.01,0.01,0.01)):
    return np.stack([np.stack([np.stack([create_psf3(N = N,
                                sig = sig)
                for _ in range(Gx)])
                for _ in range(Gy)])
                for _ in range(Gz)])


def psf_grid_linear3(Gx,Gy,Gz, N=16):
        return np.stack([np.stack([np.stack([create_psf3(N = N,
                    sig = (0.1+.4*x**2,0.001+.2*x**2,0.001+.2*x**2))
                for x in np.linspace(-1,1,Gx)])
                for y in np.linspace(-1,1,Gy)])
                for z in np.linspace(-1,1,Gz)])


def psf_grid_const3(Gx,Gy,Gz, N=21, sig = (0.01,0.01,0.01)):
    return np.stack([np.stack([np.stack([create_psf3(N = N,
                                        sig = sig)
                for _ in range(Gx)])
                for _ in range(Gy)])
                for _ in range(Gz)])


def create_prolate_psf3(N = 21, w = (0,1,0), s1=.4, s2 = .1):
    x = np.linspace(-1,1,N+1)[:-1]
    Xs = np.array(np.meshgrid(x,x,x,indexing="ij"))

    w = np.array(w)
    w = w*1./np.sqrt(np.sum(w**2))

    r1 = np.dot(Xs.T,w)
    r2 = np.sqrt(np.sum(np.cross(Xs.T,w)**2,axis = -1))

    h = np.exp(-r1**2/2./s1**2-r2**2/2./s2**2)
    h *= 1./np.sum(h)
    return h

def psf_grid_motion3(Gx=2,Gy=2,Gz=2, N = 21, s1 = .4, s2 = .1, omega = (0,1,0)):
    return np.stack([np.stack([np.stack([create_prolate_psf3(
        w = np.cross(omega, [_x,_y,_z]), N = N, s1=s1, s2 = s2)\
                             for _x in np.linspace(-1,1,Gx)])
                        for _y in np.linspace(-1,1,Gy)])
                     for _z in np.linspace(-1,1,Gz)])


def make_grid3(hs):
    Gz, Gy,Gx, Hz, Hy, Hx = hs.shape

    im = np.zeros((Gz*Hz,Gy*Hy,Gx*Hx))
    for i in range(Gx):
        for j in range(Gy):
            for k in range(Gz):
                im[k*Hz:(k+1)*Hz,j*Hy:(j+1)*Hy,i*Hx:(i+1)*Hx] = hs[k,j,i]

    return im


def test_conv2():
    im = np.zeros((128,128))
    Gx = 4
    Gy = 8
    hs = psf_grid_motion(Gx,Gy,100)
    out = convolve_spatial2(im, hs)
    return im, out, hs


def test_conv2_reflect():
    im = np.zeros((128, 128))
    Gx, Gy = 4, 8
    hs = psf_grid_motion(Gx, Gy, 100)
    out = convolve_spatial2(im, hs, mode='reflect')
    return im, out, hs


def test_conv2_psfs():
    im = np.zeros((384,512))
    im[::32, ::32] = 1.
    Gx = 8
    Gy = 4
    hs = psf_grid_motion(Gx,Gy,30)
    out = convolve_spatial2(im, hs)
    return im, out, hs


def test_conv3():
    im = np.zeros((128,64,32))
    Gx = 8
    Gy = 4
    Gz = 2
    hs = psf_grid_linear3(Gx,Gy,Gz,10)
    out = convolve_spatial3(im, hs)
    return im, out, hs


def test_conv3_reflect():
    im = np.zeros((128, 64, 32))
    Gx, Gy, Gz = 8, 4, 2
    hs = psf_grid_linear3(Gx, Gy, Gz, 10)
    out = convolve_spatial3(im, hs, mode='reflect')
    return im, out, hs


def test_conv3_psfs():
    im = np.zeros((128, 128,128))
    im[::16,::16,::16] = 1.
    Gx = 16
    Gy = 16
    Gz = 16
    hs = psf_grid_motion3(Gx,Gy,Gz,20)

    out = convolve_spatial3(im, hs)
    return im,out, hs


def speed_test3(imshape=(128,128,128), gshape=(4,4,4)):
    im = np.zeros(imshape)
    hs = np.ones(gshape+imshape)

    out, plan = convolve_spatial3(im, hs, return_plan=True)
    t  = time()
    out = convolve_spatial3(im, hs, plan=plan)
    t = time()-t
    print("imshape: %s \tgshape: %s   \ttime = %.2gs"%(imshape, gshape, t))
    return t


def test_single_z():
    Ng = 32
    Nx = 512
    Nh = 3

    im = np.zeros((16, Nx, Nx))
    im[4:-4, 4::16, 4::16] = 1.
    hs = np.zeros((16, Nx, Nx))

    for i in range(Ng):
        for j in range(Ng):
            si = slice(i*(Nx//Ng)-Nh+(Nx//Ng)//2,i*(Nx//Ng)+Nh+1+(Nx//Ng)//2)
            sj = slice(j * (Nx // Ng) - Nh+(Nx//Ng)//2, j * (Nx // Ng) + Nh+1+(Nx//Ng)//2)
            hs[:,sj,si] = np.ones((16,2*Nh+1,2*Nh+1))

    hs[:,Nx//Ng//2::Nx//Ng,Nx//Ng//2::Nx//Ng] = 1.

    out = convolve_spatial3(im, hs, grid_dim = (1, Ng,Ng), pad_factor=2)
    return im, out, hs


def test_identity2():
    from scipy.misc import ascent
    from scipy.ndimage.interpolation import zoom

    im = zoom(ascent().astype(np.float32),(2,2))

    Ng = 32
    Ny,Nx = im.shape

    h = np.zeros_like(im)
    h[Ny//Ng//2::Ny//Ng,Nx//Ng//2::Nx//Ng] = 1.

    out = convolve_spatial2(im, h, grid_dim = (Ng,Ng), pad_factor=3)

    #npt.assert_almost_equal(im, out, decimal = 3)
    return im, out, h


if __name__ == '__main__':

    #im, out, hs = test_single_z()
    # im2, out2, hs2 = test_conv2_psfs()
    # im3, out3, hs3 = test_conv3_psfs()

    # ts = [speed_test3((128,)*3,(4,4,2**n)) for n in range(5)]

    im, out, h = test_identity2()