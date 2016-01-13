"""
calculate 2d or 3d perlin noise on the gpu

mweigert@mpi-cbg.de

"""


from gputools import OCLProgram, OCLArray
import numpy as np

def abspath(myPath):
    import sys, os
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
        return os.path.join(base_path, os.path.basename(myPath))
    except Exception:
        base_path = os.path.abspath(os.path.dirname(__file__))
        return os.path.join(base_path, myPath)



def perlin2(size, repeat = (10.,)*2):
    wx, wy = repeat

    prog = OCLProgram(abspath("perlin.cl"))

    d = OCLArray.empty(size[::-1],np.float32)
    prog.run_kernel("perlin2d",d.shape[::-1],None,
                    d.data,
                    np.float32(wx),np.float32(wy))

    return d.get()


def _perlin3_single(size,repeat = (10.,)*3,offz = 0,Nz0 = None):
    if Nz0 is None:
        Nz0 = size[-1]

    wx, wy, wz = repeat

    prog = OCLProgram(abspath("perlin.cl"))

    d = OCLArray.empty(size[::-1],np.float32)
    prog.run_kernel("perlin3d",d.shape[::-1],None,
                    d.data,
                    np.int32(Nz0),
                    np.int32(offz),
                    np.float32(wx),np.float32(wy),np.float32(wz) )

    return d.get()


def perlin3(size,repeat = (10.,)*3,n_volumes=1):
    """returns a 3d perlin noise array of given size (Nx,Ny,Nz) with given repeats
    by doing the noise calculations on the gpu

    The volume can be splitted into n_volumes pieces if gou memory is not enough
    """

    if n_volumes ==1:
        return _perlin3_single(size,repeat)
    else:
        Nx, Ny, Nz = size
        Nz2 = Nz/n_volumes+1
        res = np.empty((Nz,Ny,Nx),np.float32)
        res_part = np.empty((Nz2,Ny,Nx),np.float32)
        for i in range(n_volumes):
            i1,i2 = i*Nz2, np.clip((i+1)*Nz2,0,Nz)
            if i<n_volumes-1:
                res_part = _perlin3_single((Nx,Ny,i2-i1+1),
                                    repeat,offz = Nz2*i,Nz0 = Nz)

                res[i1:i2,...] = res_part[:-1,...]
            else:
                res_part = _perlin3_single((Nx,Ny,i2-i1),
                                    repeat,offz = Nz2*i,Nz0 = Nz)

                res[i1:i2,...] = res_part
        return res





if __name__ == '__main__':
    N = 256

    ws = (10,)*3


    d1 = perlin3((N,)*3,ws, n_volumes=1)
    d2 = perlin3((N,)*3,ws, n_volumes=2)


    print np.allclose(d1,d2)
