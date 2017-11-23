"""


mweigert@mpi-cbg.de

"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from time import time
from functools import reduce
from gputools import convolve_spatial2, convolve_spatial3


def create_psf(sig=(.1,.1), xy_angle = 0., N = 10, center = (0,0)):
    x = np.linspace(-1,1,N+1)[:-1]
    Y, X = np.meshgrid(x,x,indexing="ij")
    X2 = np.cos(xy_angle)*X - np.sin(xy_angle)*Y
    Y2 = np.cos(xy_angle)*Y + np.sin(xy_angle)*X
    h = np.exp(-reduce(np.add,[(_X-_c)**2/_s**2/2. for _X,_s,_c in zip([Y2,X2],sig, center)]))
    h *= 1./np.sum(h)
    return h


def psf_grid_motion(Gx,Gy,N = 21):
    return np.stack([np.stack([create_psf(sig = (.01+.1*np.sqrt(_x**2+_y**2),
                                          .01+.4*np.sqrt(_x**2+_y**2)),
                                          N = N,
                                          xy_angle = -1.*np.pi+np.arctan2(_y,_x))\
                             for _y in np.linspace(-1,1,Gy)]) for _x in np.linspace(-1,1,Gx)])

def psf_grid_cushion(Gx,Gy,N = 21):
    return np.stack([np.stack([create_psf(sig = (.05,.1),
                                          xy_angle=np.arctan2(_x, _y),
                                          N = N, center = (.4*(_x**2-1.),.4*(_y**2-1.)))
                             for _y in np.linspace(-1,1,Gy)])  for _x in np.linspace(-1,1,Gx)])


def psf_grid_const(Gx,Gy,N=21, sx = 0.01, sy = 0.01, center = (0,0)):
    return np.stack([np.stack([create_psf(sig=(sy,sx),N = N, center = center)
                             for _y in np.linspace(-1,1,Gy)])  for _x in np.linspace(-1,1,Gx)])





def make_grid2(hs):
    Gy,Gx, Hy, Hx = hs.shape

    im = np.zeros((Gy*Hy,Gx*Hx))
    for i in range(Gx):
        for j in range(Gy):
            im[j*Hy:(j+1)*Hy,i*Hx:(i+1)*Hx] = hs[j,i]

    return im

def conv(x, hs, transp = False):
    if transp:
        return convolve_spatial2(x,hs[...,::-1,::-1])
    else:
        return convolve_spatial2(x, hs)


# lucy richardson
def lucy(y, hs, n_iter =20):
    def lucy_step(x,im0, hs):
        y = conv(x, hs)
        y = im0/(y+1.e-5)
        y = conv(y,hs, transp=True)
        return x*y
    x = y.copy()
    for i in range(n_iter):
        print(i)
        x = lucy_step(x,y, hs)
    return x

if __name__ == '__main__':

    im = np.zeros((512,512))
    im[::32] = 1.
    im[:,::32] = 1.
    Gx = 8
    Gy = 8
    hs = psf_grid_cushion(Gx,Gy,30)
    hs0 = psf_grid_const(Gx,Gy, sx=.3, sy = .3, center=(.2,.2))


    out0 = conv(im, hs0)+0.02*np.random.uniform(0,1,im.shape)
    out = conv(im, hs)+0.02*np.random.uniform(0,1,im.shape)


    u0 = lucy(out0,hs0)
    u = lucy(out, hs)



