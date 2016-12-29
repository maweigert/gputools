"""
A collection of some denoising algorithms for 3d

MW, 2014
"""


from __future__ import print_function, unicode_literals, absolute_import, division
import os
import subprocess

import numpy as np
from scipy.misc import lena, imsave, imread
from itertools import product
import utils
import imgtools
from time import time


def absPath(myPath):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
        return os.path.join(base_path, os.path.basename(myPath))
    except Exception:
        base_path = os.path.abspath(os.path.dirname(__file__))
        return os.path.join(base_path, myPath)


def bilateral3(data, fSize, sigmaX, dev = None):

    if dev is None:
        dev = imgtools.__DEFAULT_OPENCL_DEVICE__

    if dev is None:
        raise ValueError("no OpenCLDevice found...")


    dtype = data.dtype.type
    dtypes_kernels = {np.float32:"run3_float",
                        np.uint16:"run3_short"}

    if not dtype in dtypes_kernels:
        print("data type %s not supported yet, please convert to:"%dtype,list(dtypes_kernels.keys()))
        return

    proc = OCLProcessor(dev,utils.absPath("kernels/bilateral.cl"))

    clImg = dev.createImage(data.shape[::-1],channel_type = cl_datatype_dict[dtype])
    clBuf = dev.createBuffer(data.size, dtype= dtype)
    dev.writeImage(clImg,data)

    proc.runKernel(dtypes_kernels[dtype],clImg.shape,None,clImg,clBuf,
                     np.int32(clImg.shape[0]),np.int32(clImg.shape[1]),
                     np.int32(fSize),np.float32(sigmaX))

    return dev.readBuffer(clBuf,dtype=dtype).reshape(data.shape)

def wiener3(data,sigma):
    d_f = np.fft.rfftn(data)
    pf = np.abs(d_f)**2
    w_f = pf/(pf+8*data.size*sigma**2)
    return np.real(np.fft.irfftn(d_f *w_f))


def nlm3(data, fSize, bSize, sigma, dev = None):

    if dev is None:
        dev = imgtools.__DEFAULT_OPENCL_DEVICE__

    if dev is None:
        raise ValueError("no OpenCLDevice found...")

    dtype = data.dtype.type
    dtypes_kernels = {np.float32:"nlm3_float",
                        np.uint16:"nlm3_short"}

    if not dtype in dtypes_kernels:
        print("data type %s not supported yet, please convert to:"%dtype,list(dtypes_kernels.keys()))
        return

    proc = OCLProcessor(dev,utils.absPath("kernels/nlmeans3d.cl"))

    img = dev.createImage(data.shape[::-1],channel_type = cl_datatype_dict[dtype])
    buf = dev.createBuffer(data.size,dtype = dtype)
    dev.writeImage(img,data)


    print(img.shape)
    proc.runKernel(dtypes_kernels[dtype],img.shape,None,img,buf,
                     np.int32(img.width),np.int32(img.height),
                     np.int32(fSize), np.int32(bSize),np.float32(sigma))

    return dev.readBuffer(buf,dtype=dtype).reshape(data.shape)


# class DenoiseNLM3:
#     def __init__(self,dshape = (64,)*3, FS=2,BS=3, dev= None):
#         if not dev:
#             dev= OCLDevice()
#         self.dev = dev
#         self.FS, self.BS = FS,BS
#         self.proc = OCLProcessor(self.dev,utils.absPath("kernels/nlm_fast3.cl"),options="-D FS=%i -D BS=%i"%(FS,BS))
#         self.init_data(dshape)

#     def init_data(self,dshape):
#         Nz, Ny, Nx = dshape
#         self.inImg = self.dev.createImage((Nx,Ny,Nz),channel_type = cl.channel_type.FLOAT )
#         self.distImg = self.dev.createImage((Nx,Ny,Nz),
#                                   mem_flags = cl.mem_flags.READ_WRITE,
#                                   channel_type = cl.channel_type.FLOAT )

#         self.tmpImg = self.dev.createImage((Nx,Ny,Nz),
#                                      mem_flags = cl.mem_flags.READ_WRITE,
#                                      channel_type = cl.channel_type.FLOAT )
#         self.tmp2Img = self.dev.createImage((Nx,Ny,Nz),
#                                   mem_flags = cl.mem_flags.READ_WRITE,
#                                   channel_type = cl.channel_type.FLOAT )

#         self.accBuf = self.dev.createBuffer(Nx*Ny*Nz,
#                                   mem_flags = cl.mem_flags.READ_WRITE,
#                                   dtype = np.float32)

#         self.weightBuf = self.dev.createBuffer(Nx*Ny*Nz,
#                                      mem_flags = cl.mem_flags.READ_WRITE,
#                                      dtype = np.float32)

#     def run(self,data, sigma):
#         Nx,Ny,Nz = self.inImg.shape

#         self.dev.writeImage(self.inImg,data)
#         self.dev.writeBuffer(self.accBuf,np.zeros((Ny,Nx,Nz),dtype=np.float32))
#         self.dev.writeBuffer(self.weightBuf,np.zeros((Ny,Nx,Nz),dtype=np.float32))

#         for dx in range(self.BS+1):
#             for dy in range(-self.BS,self.BS+1):
#                 for dz in range(-self.BS,self.BS+1):
#                     self.proc.runKernel("dist",(Nx,Ny,Nz),None,self.inImg,self.tmpImg,
#                                    np.int32(dx),np.int32(dy),np.int32(dz))
#                     self.proc.runKernel("convolve",(Nx,Ny,Nz),None,self.tmpImg,
#                                    self.tmp2Img,np.int32(1))
#                     self.proc.runKernel("convolve",(Nx,Ny,Nz),None,self.tmp2Img,
#                                    self.tmpImg,np.int32(2))
#                     self.proc.runKernel("convolve",(Nx,Ny,Nz),None,self.tmpImg,
#                                    self.distImg,np.int32(4))

#                     self.proc.runKernel("computePlus",(Nx,Ny,Nz),None,self.inImg,
#                                    self.distImg,self.accBuf,self.weightBuf,
#                                    np.int32(Nx),np.int32(Ny),np.int32(Nz),
#                                    np.int32(dx),np.int32(dy),np.int32(dz),
#                                    np.float32(sigma))

#                     if any([dx,dy,dz]):
#                         self.proc.runKernel("computeMinus",(Nx,Ny,Nz),None,
#                                        self.inImg,self.distImg,
#                                        self.accBuf,self.weightBuf,
#                                        np.int32(Nx),np.int32(Ny),np.int32(Nz),
#                                        np.int32(dx),np.int32(dy),np.int32(dz),
#                                        np.float32(sigma))

#         acc  = self.dev.readBuffer(self.accBuf,dtype=np.float32).reshape((Nz,Ny,Nx))
#         weights  = self.dev.readBuffer(self.weightBuf,dtype=np.float32).reshape((Nz,Ny,Nx))

#         return acc/weights



def nlm3_fast(data,FS,BS,sigma,dev = None, proc = None):
    """for noise level (and FS,BS = 2,3) of sigma_0, choose sigma = 1.1*sigma_0
    """
    if dev is None:
        dev = imgtools.__DEFAULT_OPENCL_DEVICE__

    if dev is None:
        raise ValueError("no OpenCLDevice found...")

    dtype = data.dtype.type

    if not dtype in [np.float32,np.uint16]:
        print("data type %s not supported yet, please convert to:"%[np.float32,np.uint16])
        return

    if dtype == np.float32:
        proc = OCLProcessor(dev,utils.absPath("kernels/nlm_fast3.cl"),options="-D FS=%i -D BS=%i -D FLOAT"%(FS,BS))
    else:
        proc = OCLProcessor(dev,utils.absPath("kernels/nlm_fast3.cl"),options="-D FS=%i -D BS=%i "%(FS,BS))

    Nz,Ny, Nx = data.shape

    inImg = dev.createImage_like(data,
                             mem_flags = "READ_ONLY")


    distImg = dev.createImage_like(data,
                             mem_flags = "READ_WRITE")

    tmpImg =  dev.createImage_like(data,
                             mem_flags = "READ_WRITE")

    tmp2Img =  dev.createImage_like(data,
                             mem_flags = "READ_WRITE")

    accBuf = dev.createBuffer(Nx*Ny*Nz,
                             mem_flags = cl.mem_flags.READ_WRITE,
                             dtype = np.float32)

    weightBuf = dev.createBuffer(Nx*Ny*Nz,
                             mem_flags = cl.mem_flags.READ_WRITE,
                             dtype = np.float32)


    dev.writeImage(inImg,data);
    dev.writeBuffer(weightBuf,np.zeros((Ny,Nx,Nz),dtype=np.float32));

    tdist = 0
    tconv = 0
    tcomp =0

    from time import time
    for dx in range(BS+1):
        for dy in range(-BS,BS+1):
            for dz in range(-BS,BS+1):
                t = time()
                proc.runKernel("dist",(Nx,Ny,Nz),None,inImg,tmpImg,np.int32(dx),np.int32(dy),np.int32(dz)).wait()
                tdist += time()-t

                t = time()

                proc.runKernel("convolve",(Nx,Ny,Nz),None,tmpImg,tmp2Img,np.int32(1))
                proc.runKernel("convolve",(Nx,Ny,Nz),None,tmp2Img,tmpImg,np.int32(2))
                proc.runKernel("convolve",(Nx,Ny,Nz),None,tmpImg,distImg,np.int32(4)).wait()
                tconv += time()-t

                t = time()

                proc.runKernel("computePlus",(Nx,Ny,Nz),None,inImg,distImg,accBuf,weightBuf,
                               np.int32(Nx),np.int32(Ny),np.int32(Nz),
                               np.int32(dx),np.int32(dy),np.int32(dz),np.float32(sigma))

                if any([dx,dy,dz]):
                    proc.runKernel("computeMinus",(Nx,Ny,Nz),None,inImg,distImg,accBuf,weightBuf,
                               np.int32(Nx),np.int32(Ny),np.int32(Nz),
                               np.int32(dx),np.int32(dy),np.int32(dz),np.float32(sigma)).wait()

                tcomp += time()-t

    # print tdist,tconv, tcomp

    acc  = dev.readBuffer(accBuf,dtype=np.float32).reshape((Nz,Ny,Nx))
    weights  = dev.readBuffer(weightBuf,dtype=np.float32).reshape((Nz,Ny,Nx))

    return acc/weights



def test_nlm3_fast():
    from imgtools import test_images, calcPSNR, read3dTiff
    dev = OCLDevice(useDevice=1)



    # data = test_images.blobs64()

    data = read3dTiff("/Users/mweigert/Data/synthetics/blobs64.tif")

    np.random.seed(0)
    data = 1000.*data/np.amax(data)
    y = np.maximum(0,data + np.random.normal(0,200,data.shape))
    # y = y.astype(np.uint16)
    y = y.astype(np.float32)

    t = time()
    out = nlm3_fast(dev,y,2,3,200)

    print("time:", time()-t)
    print("PSNR: ",utils.calcPSNR(data,out))
    # sigs = np.linspace(3,70,30)

    # bests = []
    # for s0 in np.linspace(2,20,10):
    #     y = data + np.random.normal(0,s0,data.shape)
    #     y = y.astype(np.float32)
    #     sigs = s0*np.linspace(.1,5 ,20)

    #     ind=np.argmax([calcPSNR(data,nlm3_fast(dev,y,2,3,s)) for s in sigs])
    #     print s0,sigs[ind]
    #     bests.append([s0,sigs[ind]])

    return out




def nlm3_thresh(dev, data, FS,BS, sigma, thresh= 0, mean = False):

    dtype = data.dtype.type

    if not dtype in [np.float32,np.uint16]:
        print("data type %s not supported yet, please convert to:"%[np.float32,np.uint16])
        return

    if dtype == np.float32:
        proc = OCLProcessor(dev,utils.absPath("kernels/nlm3_thresh.cl"),options="-D FS=%i -D BS=%i -D FLOAT"%(FS,BS))
    else:
        proc = OCLProcessor(dev,utils.absPath("kernels/nlm3_thresh.cl"),options="-D FS=%i -D BS=%i"%(FS,BS))


    img = dev.createImage_like(data)
    buf = dev.createBuffer(data.size,dtype = dtype)
    dev.writeImage(img,data)


    proc.runKernel("nlm3_thresh",img.shape,None,img,buf,
                   np.int32(img.width),np.int32(img.height),
                   np.float32(sigma),np.float32(thresh))

    return dev.readBuffer(buf,dtype = dtype).reshape(data.shape)



def test_nlm3_thresh():
    from imgtools import test_images, calcPSNR, read3dTiff
    dev = OCLDevice(useDevice=1)



    # data = test_images.blobs64()

    data = read3dTiff("/Users/mweigert/Data/synthetics/blobs64.tif")

    np.random.seed(0)
    data = 100.*data/np.amax(data)
    y = np.maximum(0,data + np.random.normal(0,20,data.shape))
    y = y.astype(np.float32)
    y = y.astype(np.uint16)

    t = time()
    out = nlm3_thresh(dev,y,2,3,30,60)

    print(time()-t)
    print(calcPSNR(data,out))
    # sigs = np.linspace(3,70,30)

    # bests = []
    # for s0 in np.linspace(2,20,10):
    #     y = data + np.random.normal(0,s0,data.shape)
    #     y = y.astype(np.float32)
    #     sigs = s0*np.linspace(.1,5 ,20)

    #     ind=np.argmax([calcPSNR(data,nlm3_fast(dev,y,2,3,s)) for s in sigs])
    #     print s0,sigs[ind]
    #     bests.append([s0,sigs[ind]])

    return out



def tv3_gpu(data,weight,Niter=50, Ncut = 1, dev = None):
    """
    chambolles tv regularized denoising on the gpu

    weight should be around  2+1.5*noise_sigma
    """

    if dev is None:
        dev = imgtools.__DEFAULT_OPENCL_DEVICE__

    if dev is None:
        raise ValueError("no OpenCLDevice found...")


    proc = OCLProcessor(dev,utils.absPath("kernels/tv_chambolle.cl"))

    if Ncut ==1:
        inImg = dev.createImage(data.shape[::-1],dtype = np.float32)

        pImgs = [ dev.createImage(data.shape[::-1],
                                  mem_flags = cl.mem_flags.READ_WRITE,
                                  dtype= np.float32,
                                  channel_order = cl.channel_order.RGBA)
                                  for i in range(2)]

        outImg = dev.createImage(data.shape[::-1],
                                 dtype = np.float32,
                                 mem_flags = cl.mem_flags.READ_WRITE)


        dev.writeImage(inImg,data.astype(np.float32));
        dev.writeImage(pImgs[0],np.zeros((4,)+data.shape,dtype=np.float32));
        dev.writeImage(pImgs[1],np.zeros((4,)+data.shape,dtype=np.float32));


        for i in range(Niter):
            proc.runKernel("div_step",inImg.shape,None,
                           inImg,pImgs[i%2],outImg)
            proc.runKernel("grad_step",inImg.shape,None,
                           outImg,pImgs[i%2],pImgs[1-i%2],
                           np.float32(weight))
        return dev.readImage(outImg,dtype=np.float32)

    else:
        res = np.empty_like(data,dtype=np.float32)
        Nz,Ny,Nx = data.shape
        # a heuristic guess: Npad = Niter means perfect
        Npad = 1+Niter/2
        for i0,(i,j,k) in enumerate(product(list(range(Ncut)),repeat=3)):
            print("calculating box  %i/%i"%(i0+1,Ncut**3))
            sx = slice(i*Nx/Ncut,(i+1)*Nx/Ncut)
            sy = slice(j*Ny/Ncut,(j+1)*Ny/Ncut)
            sz = slice(k*Nz/Ncut,(k+1)*Nz/Ncut)
            sx1,sx2 = utils._extended_slice(sx,Nx,Npad)
            sy1,sy2 = utils._extended_slice(sy,Ny,Npad)
            sz1,sz2 = utils._extended_slice(sz,Nz,Npad)

            data_sliced = data[sz1,sy1,sx1].copy()
            _res = tv3_gpu(dev,data_sliced,weight,Niter,Ncut = 1)
            res[sz,sy,sx] = _res[sz2,sy2,sx2]

        return res

def test_tv3_gpu():
    from imgtools import test_images, calcPSNR, read3dTiff
    dev = OCLDevice(useDevice=1)

    data = read3dTiff("/Users/mweigert/Data/synthetics/blobs64.tif")

    np.random.seed(0)
    data = 100.*data/np.amax(data)
    y = np.maximum(0,data + np.random.normal(0,0.2*np.amax(data),data.shape))
    y = y.astype(np.float32)

    t = time()
    out = tv3_gpu(dev,y,.4*np.amax(data))

    print("time:", time()-t)
    print("PSNR: ",utils.calcPSNR(data,out))
    return data,y,out

def bm4d(data,sigma):
    tmpName = utils.absPath("0123456789_TMP.tiff")
    tmpName2 = utils.absPath("0123456789_OUT_TMP.tiff")
    imgtools.write3dTiff(data,tmpName)
    subprocess.call([utils.absPath("cxx_code/bm4d/bm4d.sh"),tmpName,tmpName2,str(sigma)], stdout=subprocess.PIPE)

    out = imgtools.read3dTiff(tmpName2)
    # os.remove(tmpName)
    # os.remove(tmpName2)
    return out

def test_bm4d():
    from imgtools import test_images, calcPSNR, read3dTiff
    dev = OCLDevice()

    data = test_images.blobs64()

    data = read3dTiff("/Users/mweigert/Data/synthetics/filaments64.tif")

    data = 100.*data/np.amax(data)
    y = data + np.random.normal(0,20,data.shape)
    y = y.astype(np.float32)

    out = bm4d(y,20)
    print(calcPSNR(data,out))
    sigs = np.linspace(3,70,30)

    return data,out


# def test_filter():
#     from functools import partial
#     import time

#     dev = OCLDevice()
#     dev.initCL(useDevice=1)
#     data = lena()
#     data = np.random.uniform(0,1,[2**9,2**9])

#     print "running on image shape : {} \n\n".format(data.shape)

#     fs = {
#         "bilateral":lambda:bilateral(dev,data,3,10,10),
#         "nlMeans":lambda:nlMeans(dev,data,3,7,10),
#         "nlMeans S":lambda:nlMeansShared(dev,data,3,7,10),
#         "nlMeans P":lambda:nlMeansProjected(dev,data,3,7,10),
#         "BM3D    ":lambda:bm3d(data,10),
#         "dct    ":lambda:dct_denoising(data,10),
#         "dct_8x8 ":lambda:dct_8x8(dev,data,10),

#     # "nlMeans prefetched":lambda:nlMeansPrefetch(dev,data,3,5,10),
#           #"nlMeans test":lambda:nlMeansTest(dev,data,3,5,10),
#           }

#     t = time.time()
#     for f in fs.keys():
#         fs[f]()
#         print "%s \t: \t %.2f s"%(f,time.time()-t)
#         t = time.time()



def test_nlm3():
    dev = OCLDevice()
    data = np.linspace(0,100,32**3).astype(np.float32).reshape((32,32,32))
    data+= np.random.normal(0,20,data.shape)
    data = data.astype(np.float32)
    out = nlm3(dev,data,2,3,20)
    return out
    # data = np.ones((64,64,64),dtype=np.uint16)
    # out = nlm3(dev,data,2,3,20)


def test_all():
    from time import time


    d = np.ones((128,)*3,np.float32)

    filters = {"nlm3_fast":lambda x:nlm3_fast(x,2,3,2.),
               "tv3_gpu":lambda x:tv3_gpu(x,3.),
               "bilateral3":lambda x:bilateral3(x,3,3),
               "wiener3":lambda x:wiener3(x,3.)
               }

    for name, filter in filters.items():
        t = time()
        list(filter(d));
        print("running time = %2.f ms \t name = %s \t size = %s "%(1000.*(time()-t),name,d.shape))




if __name__ == '__main__':


    test_all()

    # out = test_nlm3_thresh()

    # out = test_nlm3_fast()
    # data, out = test_bm4d()
    # out = test_tv3_gpu()

    # from imgtools import test_images, calcPSNR, read3dTiff

    # data = read3dTiff("/Users/mweigert/Data/synthetics/blobs256.tif")

    # np.random.seed(0)
    # data = 100+100.*data/np.amax(data)

    # y = np.maximum(0,data + np.random.normal(0,10,data.shape))
    # y = y.astype(np.float32)

    # out1 = tv3_gpu(y,17.)
    # out2 = tv3_gpu(dev,y,17.,Ncut=2)


    # sigs = np.linspace(.01,.4,10)
    # best_lam = []
    # for s in sigs:
    #     print s
    #     y = np.maximum(0,data + np.random.normal(0,s*np.amax(data),data.shape))
    #     y = y.astype(np.float32)
    #     lams = np.linspace(.1,80,40)
    #     res = [utils.calcPSNR(data,tv3_gpu(dev,y,lam)) for lam in lams]
    #     print max(res)
    #     best_lam.append(lams[np.argmax(res)])
