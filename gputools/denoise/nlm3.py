""" non local means filter

fast implementation see (fix ref)


"""
import logging
logger = logging.getLogger(__name__)


import numpy as np

from gputools import OCLArray,OCLImage, OCLProgram, get_device

from _abspath import abspath


def nlm3(data,sigma, size_filter = 2, size_search = 3):
    """for noise level of sigma_0, choose sigma = 1.5*sigma_0
    """

    prog = OCLProgram(abspath("kernels/nlm3.cl"),
                      build_options="-D FS=%i -D BS=%i"%(size_filter,size_search))


    data = data.astype(np.float32, copy = False)
    img = OCLImage.from_array(data)

    distImg = OCLImage.empty_like(data)

    distImg = OCLImage.empty_like(data)
    tmpImg = OCLImage.empty_like(data)
    tmpImg2 = OCLImage.empty_like(data)

    accBuf = OCLArray.zeros(data.shape,np.float32)    
    weightBuf = OCLArray.zeros(data.shape,np.float32)

    for dx in range(size_search+1):
        for dy in range(-size_search,size_search+1):
            for dz in range(-size_search,size_search+1):
                prog.run_kernel("dist",img.shape,None,
                                img,tmpImg,np.int32(dx),np.int32(dy),np.int32(dz))
                
                prog.run_kernel("convolve",img.shape,None,
                                tmpImg,tmpImg2,np.int32(1))
                prog.run_kernel("convolve",img.shape,None,
                                tmpImg2,tmpImg,np.int32(2))
                prog.run_kernel("convolve",img.shape,None,
                                tmpImg,distImg,np.int32(4))

                prog.run_kernel("computePlus",img.shape,None,
                                img,distImg,accBuf.data,weightBuf.data,
                                np.int32(img.shape[0]),
                                np.int32(img.shape[1]),
                                np.int32(img.shape[2]),
                                np.int32(dx),np.int32(dy),np.int32(dz),
                                np.float32(sigma))

                if any([dx,dy,dz]):
                    prog.run_kernel("computeMinus",img.shape,None,
                                    img,distImg,accBuf.data,weightBuf.data,
                                    np.int32(img.shape[0]),
                                    np.int32(img.shape[1]),
                                    np.int32(img.shape[2]),
                                    np.int32(dx),np.int32(dy),np.int32(dz),
                                    np.float32(sigma))

    acc  = accBuf.get()
    weights  = weightBuf.get()

    return acc/weights



if __name__ == '__main__':
    d = 10*np.linspace(0,1,31*32*33).reshape((31,32,33))

    d += np.random.normal(0,1,d.shape)

    res = nlm3(d,100,2,3)
