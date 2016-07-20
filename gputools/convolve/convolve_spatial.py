"""
spatially varying convolutions


mweigert@mpi-cbg.de

"""


import numpy as np
from gputools import fft_plan, OCLArray, OCLImage, fft, get_device, OCLProgram, pad_to_shape
from gputools.utils.utils import _is_power2, _next_power_of_2
from _abspath import abspath

def convolve_spatial2(im, hs, plane = None, return_plan = False):
    """
    spatial varying convolution of an 2d image with a 2d grid of psfs

    shape(im_ = (Ny,Nx)
    shape(hs) = (Gy,Gx,Hy,Hx)

    the psfs are assumed to be defined equally spaced
    i.e. hs[0,0] is at (0,0) and hs[-1,-1] at (Ny-1,Nx-1)
    """
    if not np.all([_is_power2(n) for n in im.shape]):
        raise NotImplementedError("im.shape == %s has to be power of 2 as of now"%(str(im.shape)))

    if not np.all([n%(g-1)==0 for n,g in zip(im.shape,hs.shape[:2])]):
        raise NotImplementedError("Gx Gy  = %s shape mismatch"%(str(hs.shape)))

    Ny, Nx = im.shape
    Gy, Gx = hs.shape[:2]

    # the size of each block within the grid
    Nblock_y, Nblock_x = Ny/(Gy-1), Nx/(Gx-1)


    # the size of the overlapping patches with safety padding
    Npatch_x, Npatch_y = _next_power_of_2(3*Nblock_x), _next_power_of_2(3*Nblock_y)

    hs = np.fft.fftshift(pad_to_shape(hs,(Gy,Gx,Npatch_y,Npatch_x)),axes=(2,3))

    x0s = np.linspace(0,Nx-1,Gx).astype(int)
    y0s = np.linspace(0,Ny-1,Gy).astype(int)

    prog = OCLProgram(abspath("kernels/conv_spatial.cl"))

    if plane is None:
        plan = fft_plan((Npatch_x,Npatch_y))


    patches_g = OCLArray.empty((Gy,Gx,Npatch_y,Npatch_x),np.complex64)

    h_g = OCLArray.from_array(hs.astype(np.complex64))

    im_g = OCLImage.from_array(im.astype(np.float32,copy=False))
    for i,_x0 in enumerate(x0s):
        for j,_y0 in enumerate(y0s):
            prog.run_kernel("fill_patch2",(Npatch_x,Npatch_y),None,
                    im_g,np.int32(_x0-Npatch_x/2),np.int32(_y0-Npatch_y/2),
                    patches_g.data,np.int32(i*Npatch_x*Npatch_y+j*Gx*Npatch_x*Npatch_y))

    # im_g = OCLArray.from_array(im.astype(np.float32,copy=False))
    # for i,_x0 in enumerate(x0s):
    #     for j,_y0 in enumerate(y0s):
    #         prog.run_kernel("fill_patch_2d2",(Npatch_x,Npatch_y),None,
    #                     im_g.data,
    #                     np.int32(Nx),np.int32(Ny),
    #                     np.int32(_x0-Npatch_x/2),np.int32(_y0-Npatch_y/2),
    #                     patches_g.data,np.int32(i*Npatch_x*Npatch_y+j*Gx*Npatch_x*Npatch_y))

    # convolution
    fft(patches_g,inplace=True, batch = Gx*Gy, plan = plan)
    fft(h_g,inplace=True, batch = Gx*Gy, plan = plan)
    patches_g = patches_g *h_g

    fft(patches_g,inplace=True, inverse = True, batch = Gx*Gy, plan = plan)

    #accumulate
    res_g = OCLArray.empty(im.shape,np.float32)

    for i in xrange(Gx-1):
        for j in xrange(Gy-1):
            prog.run_kernel("interpolate2",(Nblock_x,Nblock_y),None,
                            patches_g.data,res_g.data,
                            np.int32(i),np.int32(j),
                            np.int32(Gx),np.int32(Gy),
                            np.int32(Npatch_x),np.int32(Npatch_y))


    res = res_g.get()

    if return_plan:
        return res, plan
    else:
        return res


def convolve_spatial3(im, hs, plan = None, return_plan = False):
    """
    spatial varying convolution of an 3d image with a 2d grid of psfs

    shape(im_ = (Nz,Ny,Nx)
    shape(hs) = (Gy,Gx, Hz,Hy,Hx)

    the psfs are assumed to be defined equally spaced
    i.e. hs[0,0] is at (0,0) and hs[-1,-1] at (Ny-1,Nx-1)
    """

    if not np.all([_is_power2(n) for n in im.shape]):
        raise NotImplementedError("im.shape == %s has to be power of 2 as of now"%(str(im.shape)))


    if not np.all([n%(g-1)==0 for n,g in zip(im.shape[:2],hs.shape[:2])]):
        raise NotImplementedError("Gx Gy  = %s shape mismatch"%(str(hs.shape)))

    Nz, Ny, Nx = im.shape
    Gy, Gx = hs.shape[:2]

    # the size of each block within the grid
    Nblock_y, Nblock_x = Ny/(Gy-1), Nx/(Gx-1)


    # the size of the overlapping patches with safety padding
    Npatch_x, Npatch_y = _next_power_of_2(3*Nblock_x), _next_power_of_2(3*Nblock_y)

    hs = np.fft.fftshift(pad_to_shape(hs,(Gy,Gx,Nz,Npatch_y,Npatch_x)),axes=(2,3,4))

    x0s = np.linspace(0,Nx-1,Gx).astype(int)
    y0s = np.linspace(0,Ny-1,Gy).astype(int)


    prog = OCLProgram(abspath("kernels/conv_spatial.cl"))

    if plan is None:
        plan = fft_plan((Nz,Npatch_x,Npatch_y))

    patches_g = OCLArray.empty((Gy,Gx,Nz,Npatch_y,Npatch_x),np.complex64)

    h_g = OCLArray.from_array(hs.astype(np.complex64))

    im_g = OCLImage.from_array(im.astype(np.float32,copy=False))

    x0s = np.linspace(0,Nx-1,Gx).astype(int)
    y0s = np.linspace(0,Ny-1,Gy).astype(int)

    for i,_x0 in enumerate(x0s):
        for j,_y0 in enumerate(y0s):
            prog.run_kernel("fill_patch3",(Npatch_x,Npatch_y,Nz),None,
                    im_g,np.int32(_x0-Npatch_x/2),np.int32(_y0-Npatch_y/2),
                    patches_g.data,np.int32(i*Nz*Npatch_x*Npatch_y+j*Gx*Nz*Npatch_x*Npatch_y))


    # convolution
    fft(patches_g,inplace=True, batch = Gx*Gy, plan = plan)
    fft(h_g,inplace=True, batch = Gx*Gy, plan = plan)
    patches_g = patches_g *h_g
    fft(patches_g,inplace=True, inverse = True, batch = Gx*Gy, plan = plan)


    #accumulate
    res_g = OCLArray.empty(im.shape,np.float32)

    for i in xrange(Gx-1):
        for j in xrange(Gy-1):
            prog.run_kernel("interpolate3",(Nblock_x,Nblock_y,Nz),None,
                            patches_g.data,res_g.data,
                            np.int32(i),np.int32(j),
                            np.int32(Gx),np.int32(Gy),
                            np.int32(Npatch_x),np.int32(Npatch_y))

    res = res_g.get()

    if return_plan:
        return res, plan
    else:
        return res



if __name__ == '__main__':
    pass
