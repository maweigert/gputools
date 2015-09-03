import numpy as np
from imgtools.ocl_fft import ocl_fft, ocl_ifft, ocl_convolve

def blur_psf(size ,radius=2, mode="gaussian"):
    """ creates a psf kernel with given radius and size
    mode = "gaussian", "disk"

    returns
       psf

       with shape(psf) = size
    """

    # make it assymmetric so that dc component is right
    xs = [N/2.*np.linspace(-1,1,N+1)[:-1] for N in size]
    if len(size) ==1:
        Xs = xs
    else:
        Xs = np.meshgrid(*xs,indexing="ij")


    if mode=="gaussian":
        psf = np.exp(-1.*np.sum(x**2 for x in Xs)/.5/radius**2)
    elif mode =="disk":
        psf = 1.*(np.sum(x**2 for x in Xs)<radius**2)
    else:
        raise NotImplementedError("mode '%s' not defined"%mode)

    psf *= 1./np.sum(psf)

    return psf

def blur(data,radius=2.,mode="gaussian",return_psf = False, gpu = False):
    """ blurs data with a kernel of size radius
    mode = "gaussian", "disk"

    if return_psf

    returns
       blurimage, psf

    else
       blurImage

    """

    psf = blur_psf(data.shape,radius,mode)

    if gpu:
        data_blurred = np.abs(ocl_convolve(data,np.fft.fftshift(psf)))

    else:
        psf_f = np.fft.rfftn(np.fft.fftshift(psf))
        data_f = np.fft.rfftn(data)
        data_blurred = np.real(np.fft.irfftn(data_f*psf_f))
        
    if return_psf:
        return data_blurred, psf

    else:
        return data_blurred



def test_blur():

    d = np.ones((100,100))
    y = blur(d,2,"gaussian")
    y = blur(d,5,"disk")

    y, psf = blur(d,2,"gaussian", return_psf= True)
    y, psf = blur(d,2,"disk", return_psf = True)




if __name__ == "__main__":

    from imgtools import test_images

    data = test_images.mpi_logo3()

    a,b = blur(data,3,mode="disk", return_psf = True)
