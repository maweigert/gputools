"""


mweigert@mpi-cbg.de

"""
import numpy as np
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


def psf_grid_const3(Gx,Gy,N=21, sig = (0.01,0.01,0.01)):
    return np.stack([np.stack([create_psf3(N = N,
                                        sig = sig)
                             for _y in np.linspace(-1,1,Gy)])  for _x in np.linspace(-1,1,Gx)])

def psf_grid_linear3(Gx,Gy,N=16):
        return np.stack([np.stack([create_psf3(N = N,
                                        sig = (0.1+.4*_x**2,0.001+.2*_x**2,0.001+.2*_x**2))
                             for _x in np.linspace(-1,1,Gx)])  for _y in np.linspace(-1,1,Gy)])


def make_grid2(hs):
    Gy,Gx, Hy, Hx = hs.shape

    im = np.zeros((Gy*Hy,Gx*Hx))
    for i in xrange(Gx):
        for j in xrange(Gy):
            im[j*Hy:(j+1)*Hy,i*Hx:(i+1)*Hx] = hs[j,i]

    return im

def make_grid3(hs):
    Gy,Gx, Hz, Hy, Hx = hs.shape

    im = np.zeros((Hz, Gy*Hy,Gx*Hx))
    for i in xrange(Gx):
        for j in xrange(Gy):
            im[:, j*Hy:(j+1)*Hy,i*Hx:(i+1)*Hx] = hs[j,i]

    return im


def test_conv2():
    from imgtools import test_images
    im = test_images.lena().astype(np.float32)
    Gx = 16+1
    hs = psf_grid_motion(Gx,Gx,100)
    out = convolve_spatial2(im, hs)
    return out, hs

def test_conv3():
    from imgtools import test_images
    im = test_images.droso128().astype(np.float32)
    Gx = 16+1
    hs = psf_grid_linear3(Gx,Gx,50)
    out = convolve_spatial3(im, hs)
    return out, hs


if __name__ == '__main__':

    out2, hs2 = test_conv2()
    out3, hs3 = test_conv3()