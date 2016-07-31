
import numpy as np
import gputools


def test_conv_gpu():

    N  = 128
    d = np.zeros((N,N+3,N+5),np.float32)

    d[N/2,N/2,N/2]  = 1.

    h = np.exp(-10*np.linspace(-1,1,17)**2)

    res = gputools.convolve_sep3(d,h,h,h)



def test_conv_sep2_numpy():
    Nx, Ny  = 128, 200

    d = np.zeros((Ny,Nx),np.float32)

    d[::10,::10] = 1.


    hx = np.ones(8)
    hy = np.ones(3)

    res1 = gputools.convolve_sep2(d,hx,hy, sub_blocks=(1,1))
    res2 = gputools.convolve_sep2(d,hx,hy, sub_blocks=(2,11))


    assert np.allclose(res1,res2)
    return res1, res2

def test_conv_sep3_numpy():
    Nz, Nx, Ny  = 128, 203, 303

    d = np.zeros((Nz, Ny,Nx),np.float32)

    d[::10,::10,::10] = 1.


    hx = np.ones(8)
    hy = np.ones(3)
    hz = np.ones(11)


    res1 = gputools.convolve_sep3(d,hx,hy,hz, sub_blocks=(1,1,1))
    res2 = gputools.convolve_sep3(d,hx,hy,hz,  sub_blocks=(7,4,3))

    assert np.allclose(res1,res2)
    return res1, res2

if __name__ == '__main__':

    #res1, res2 = test_conv_sep2_numpy()
    res1, res2 = test_conv_sep3_numpy()
    
