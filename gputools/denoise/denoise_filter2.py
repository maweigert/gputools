"""
A collection of some denoising algorithms in 2d

MW, 2014
"""

from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import subprocess



def bilateral(data, fSize, sigma, sigma_x = 10., dev= None):
    """bilateral filter """

    if dev is None:
        dev = imgtools.__DEFAULT_OPENCL_DEVICE__

    if dev is None:
        raise ValueError("no OpenCLDevice found...")

    dtype = data.dtype.type
    dtypes_kernels = {np.float32:"run2d_float",
                        np.uint16:"run2d_short"}

    if not dtype in dtypes_kernels:
        print("data type %s not supported yet, casting to float:"%dtype,list(dtypes_kernels.keys()))
        return


    img = dev.createImage_like(data)
    buf = dev.createBuffer(data.size,dtype = dtype)
    dev.writeImage(img,data)


    proc = OCLProcessor(dev,absPath("kernels/bilateral.cl"))


    proc.runKernel(dtypes_kernels[dtype],img.shape,None,img,buf,
                     np.int32(img.shape[0]),np.int32(img.shape[1]),
                     np.int32(fSize),np.float32(sigma_x),np.float32(sigma))


    return dev.readBuffer(buf,dtype=dtype).reshape(data.shape)



def test_bilateral():
    from imgtools import test_images, calcPSNR

    data = test_images.lena()
    data = 100.*data/np.amax(data)
    y = data + np.random.normal(0,20,data.shape)
    y = y.astype(np.float32)

    outs  = [calcPSNR(data,bilateral(y,2,sigs)) for sigs in np.linspace(1.,200,20)]

    return outs




# def bilateralBuffer(clDev, data, fSize, sigmaX, sigmaP):
#     clProc = OCLProcessor(clDev,absPath("kernels/bilateral.cl"))

#     clBufIn = clDev.createBuffer(data.size, mem_flag = cl.mem_flags.READ_ONLY)
#     clBufOut = clDev.createBuffer(data.size)
#     clDev.writeBuffer(clBufIn,data.astype(np.uint16))

#     clProc.runKernel("run2dBuf",data.shape[::-1],None,clBufIn,clBufOut,
#                      np.int32(data.shape[1]),np.int32(data.shape[0]),
#                      np.int32(fSize),np.float32(sigmaX),np.float32(sigmaP))

#     return clDev.readBuffer(clBufOut).reshape(data.shape)


# def bilateralAdapt(clDev, data, sensor, fSize, sigmaX, fac):
#     clProc = OCLProcessor(clDev, absPath("kernels/bilateralAdapt.cl"))

#     clImg = clDev.createImage(data.shape[::-1])
#     clImgSensor = clDev.createImage(sensor.shape[::-1])

#     clBuf = clDev.createBuffer(data.size)
#     clDev.writeImage(clImg,data.astype(np.uint16))

#     clDev.writeImage(clImgSensor,sensor.astype(np.uint16))

#     clProc.runKernel("run2d",clImg.shape,None,clImg,clImgSensor,clBuf,
#                      np.int32(clImg.shape[0]),np.int32(clImg.shape[1]),
#                      np.int32(fSize),np.float32(sigmaX),np.float32(fac))

#     return clDev.readBuffer(clBuf).reshape(data.shape)



# def nlMeansShared(clDev, data, fSize, bSize, sigma):
#     locSize = 16

#     clProc = OCLProcessor(clDev,absPath("kernels/nlmeans.cl"),options=["-D FS=%d -D BS=%d -D GS=%d"%(fSize,bSize,locSize)])


#     clImg = clDev.createImage(data.shape[::-1])
#     clBuf = clDev.createBuffer(data.size)
#     clDev.writeImage(clImg,data.astype(np.uint16))

#     clProc.runKernel("run2d_SHARED",clImg.shape,(locSize,locSize),clImg,clBuf,
#                      np.int32(clImg.shape[0]),np.int32(clImg.shape[1]),np.float32(sigma))

#     return clDev.readBuffer(clBuf).reshape(data.shape)


def nlm(data, fSize, bSize, sigma, dev = None, proc = None):

    if dev is None:
        dev = imgtools.__DEFAULT_OPENCL_DEVICE__

    if dev is None:
        raise ValueError("no OpenCLDevice found...")

    dtype = data.dtype.type
    dtypes_kernels = {np.float32:"run2d_float",
                        np.uint16:"run2d_short"}

    if not dtype in dtypes_kernels:
        print("data type %s not supported yet, please convert to:"%dtype,list(dtypes_kernels.keys()))
        return


    img = dev.createImage_like(data)
    buf = dev.createBuffer(data.size,dtype = dtype)
    dev.writeImage(img,data)


    if proc is None:
        proc = OCLProcessor(dev,absPath("kernels/nlmeans.cl"))


    proc.runKernel(dtypes_kernels[dtype],img.shape,None,img,buf,
                     np.int32(img.shape[0]),np.int32(img.shape[1]),
                     np.int32(fSize), np.int32(bSize),np.float32(sigma)).wait()

    return dev.readBuffer(buf,dtype=dtype).reshape(data.shape)


def nlm_fast(data,FS,BS,sigma,dev = None, proc = None):
    """for noise level (and FS,BS = 2,3) of sigma_0, choose sigma = 1.5*sigma_0
    """

    if dev is None:
        dev = imgtools.__DEFAULT_OPENCL_DEVICE__

    if dev is None:
        raise ValueError("no OpenCLDevice found...")

    if proc is None:
        proc = OCLProcessor(dev,absPath("kernels/nlm_fast.cl"),options="-D FS=%i -D BS=%i"%(FS,BS))

    img = dev.createImage_like(data)

    distImg = dev.createImage_like(data)

    distImg = dev.createImage_like(data, mem_flags = "READ_WRITE")
    tmpImg = dev.createImage_like(data, mem_flags = "READ_WRITE")
    tmpImg2 = dev.createImage_like(data, mem_flags = "READ_WRITE")

    accBuf = dev.createBuffer(data.size,
                             mem_flags = cl.mem_flags.READ_WRITE,
                             dtype = np.float32)

    weightBuf = dev.createBuffer(data.size,
                             mem_flags = cl.mem_flags.READ_WRITE,
                             dtype = np.float32)


    dev.writeImage(img,data);
    dev.writeBuffer(weightBuf,np.zeros_like(data,dtype=np.float32));

    for dx in range(BS+1):
        for dy in range(-BS,BS+1):
                proc.runKernel("dist",img.shape,None,img,tmpImg,np.int32(dx),np.int32(dy))
                proc.runKernel("convolve",img.shape,None,tmpImg,tmpImg2,np.int32(1))
                proc.runKernel("convolve",img.shape,None,tmpImg2,distImg,np.int32(2))

                proc.runKernel("computePlus",img.shape,None,img,distImg,accBuf,weightBuf,
                               np.int32(img.shape[0]),np.int32(img.shape[1]),
                               np.int32(dx),np.int32(dy),np.float32(sigma))

                if any([dx,dy]):
                    proc.runKernel("computeMinus",img.shape,None,img,distImg,accBuf,weightBuf,
                               np.int32(img.shape[0]),np.int32(img.shape[1]),
                               np.int32(dx),np.int32(dy),np.float32(sigma))

    acc  = dev.readBuffer(accBuf,dtype=np.float32).reshape(data.shape)
    weights  = dev.readBuffer(weightBuf,dtype=np.float32).reshape(data.shape)

    return acc/weights


def test_nlm():
    from imgtools import test_images, calcPSNR

    data = test_images.lena()
    data = 100.*data/np.amax(data)
    y = data + np.random.normal(0,20,data.shape)
    y = y.astype(np.float32)

    out = nlm(y,2,3,25.)
    sigs = np.linspace(2,70,10)

    outs  = [calcPSNR(data,nlm(y,2,3,s)) for s in sigs]
    print("nlm: sig_max = %s" %sigs[np.argmax(outs)])

    return outs


def test_nlm_fast():
    from imgtools import test_images, calcPSNR

    data = test_images.lena()
    data = 100.*data/np.amax(data)
    y = data + np.random.normal(0,20,data.shape)
    y = y.astype(np.float32)

    out = nlm_fast(y,2,3,5.)
    sigs = np.linspace(3,70,30)
    outs  = [calcPSNR(data,nlm_fast(y,2,3,s)) for s in sigs]


    bests = []
    for s0 in np.linspace(2,20,10):
        y = data + np.random.normal(0,s0,data.shape)
        y = y.astype(np.float32)
        ind=np.argmax([calcPSNR(data,nlm_fast(y,2,3,s)) for s in sigs])
        print(ind)
        bests.append([s0,sigs[ind]])

    print("nlm fast: sig_max = %s" %sigs[np.argmax(outs)])
    return bests





# def nlMeansBuffer(clDev, data, fSize, bSize, sigma):
#     clProc = OCLProcessor(clDev,absPath("kernels/nlmeans.cl"))

#     clBufIn = clDev.createBuffer(data.size, mem_flag = cl.mem_flags.READ_ONLY)
#     clBufOut = clDev.createBuffer(data.size)
#     clDev.writeBuffer(clBufIn,data.astype(np.uint16))


#     clProc.runKernel("run2dBuf",(data.size,1),None,clBufIn,clBufOut,
#                      np.int32(data.shape[1]),np.int32(data.shape[0]),
#                      np.int32(fSize), np.int32(bSize),np.float32(sigma))

#     return clDev.readBuffer(clBufOut).reshape(data.shape)



# def nlMeansPrefetch(clDev, data, fSize, bSize, sigma):
#     clProc = OCLProcessor(clDev,absPath("kernels/nlmeans.cl"),options=["-D FS=%d -D BS=%d"%(fSize,bSize)])

#     clImg = clDev.createImage(data.shape[::-1])
#     clBuf = clDev.createBuffer(data.size)
#     clDev.writeImage(clImg,data.astype(np.uint16))


#     clProc.runKernel("run2d_FIXED",clImg.shape,None,clImg,clBuf,
#                      np.int32(clImg.shape[0]),np.int32(clImg.shape[1]),np.float32(sigma))

#     return clDev.readBuffer(clBuf).reshape(data.shape)


# def nlMeansProjected(clDev, data, fSize, bSize, sigma, proc = None):

#     clImg = clDev.createImage(data.shape[::-1])

#     patchImg = clDev.createImage(data.shape[::-1], mem_flags = cl.mem_flags.READ_WRITE, channel_order = cl.channel_order.RGBA, channel_type=cl.channel_type.FLOAT)

#     clBuf = clDev.createBuffer(data.size)

#     if proc ==None:
#         proc = OCLProcessor(clDev,absPath("kernels/patch_kernel.cl"))

#     clDev.writeImage(clImg,data.astype(np.uint16))


#     proc.runKernel("project4",clImg.shape,None,clImg,patchImg,
#                      np.int32(clImg.shape[0]),np.int32(clImg.shape[1]),np.int32(fSize))


#     proc.runKernel("nlm2dProject",clImg.shape,None,clImg,patchImg,clBuf,
#                      np.int32(clImg.shape[0]),np.int32(clImg.shape[1]),
#                      np.int32(fSize), np.int32(bSize),np.float32(sigma))

#     return clDev.readBuffer(clBuf).reshape(data.shape)


# def nlMeansProjected2(clDev, data, fSize, bSize, sigma, proc = None):

#     clImg = clDev.createImage(data.shape[::-1])

#     patchImg = clDev.createImage(data.shape[::-1], mem_flags = cl.mem_flags.READ_WRITE, channel_order = cl.channel_order.RGBA, channel_type=cl.channel_type.FLOAT)

#     clBuf = clDev.createBuffer(data.size)

#     if proc ==None:
#         proc = OCLProcessor(clDev,absPath("kernels/nlmeans_projected.cl"))

#     clDev.writeImage(clImg,data.astype(np.uint16))

#     from time import time

#     t = time()

#     proc.runKernel("project4",clImg.shape,None,clImg,patchImg,
#                      np.int32(clImg.shape[0]),np.int32(clImg.shape[1]),np.int32(fSize))


#     proc.runKernel("nlm2dProject",clImg.shape,None,clImg,patchImg,clBuf,
#                      np.int32(clImg.shape[0]),np.int32(clImg.shape[1]),
#                      np.int32(fSize), np.int32(bSize),np.float32(sigma))

#     out= clDev.readBuffer(clBuf).reshape(data.shape)
#     print time()-t ,"second"
#     return out



# def nlMeansProjectedSensor(clDev, data, sensor, fSize, bSize, sigma, proc = None):

#     clImg = clDev.createImage(data.shape[::-1])

#     patchImg = clDev.createImage(data.shape[::-1], mem_flags = cl.mem_flags.READ_WRITE, channel_order = cl.channel_order.RGBA, channel_type=cl.channel_type.FLOAT)

#     sensorImg = clDev.createImage(data.shape[::-1], mem_flags = cl.mem_flags.READ_WRITE, channel_order = cl.channel_order.R, channel_type=cl.channel_type.FLOAT)

#     clBuf = clDev.createBuffer(data.size)


#     if proc ==None:
#         proc = OCLProcessor(clDev,absPath("kernels/patch_kernel.cl"))

#     clDev.writeImage(clImg,data.astype(np.uint16))

#     clDev.writeImage(sensorImg,sensor.astype(np.float32))


#     proc.runKernel("project4",clImg.shape,None,clImg,patchImg,
#                      np.int32(clImg.shape[0]),np.int32(clImg.shape[1]),np.int32(fSize))


#     proc.runKernel("nlm2dProjectSensor",clImg.shape,None,clImg,patchImg,sensorImg,clBuf,
#                      np.int32(clImg.shape[0]),np.int32(clImg.shape[1]),
#                      np.int32(fSize), np.int32(bSize),np.float32(sigma))

#     return clDev.readBuffer(clBuf).reshape(data.shape)




def bm3d(data,sigma):
    from scipy.misc import imsave, imread

    # meanData = np.mean(data)
    fName = "0123456789_TMP.png"
    imsave(fName,data)
    subprocess.call([os.path.join(os.path.dirname(__file__),
                       "cxx_code/bm3d/bm3d"),fName,str(sigma),fName], stdout=subprocess.PIPE)
    out = imread(fName)
    return out
    # return out*(1.*meanData/np.mean(out))


def dct_denoising(data,sigma):
    from scipy.misc import imsave, imread

    meanData = np.mean(data)
    fName = "0123456789_TMP.png"
    imsave(fName,data)
    subprocess.call([os.path.join(os.path.dirname(__file__),
                       "cxx_code/dct/demo_DCTdenoising"),fName,str(sigma),fName])
    out = imread(fName)
    return out*(1.*meanData/np.mean(out))




def roundUp(n,k):
    #rounds to the next number divisible by k
    return np.int(np.ceil(1.*n/k)*k)

def roundDown(n,k):
    #rounds to the next number divisible by k
    return np.int(np.floor(1.*n/k)*k)


def dct_8x8(data,sigma,dev = None, proc=None):

    if dev is None:
        dev = imgtools.__DEFAULT_OPENCL_DEVICE__

    if dev is None:
        raise ValueError("no OpenCLDevice found...")

    if not proc:
        proc = OCLProcessor(dev,absPath("kernels/dct_8x8.cl"))


    src = dev.createImage(data.shape[::-1],channel_type = cl.channel_type.FLOAT,mem_flags = cl.mem_flags.READ_ONLY)
    acc = dev.createBuffer(roundUp(data.shape[0],8)*roundUp(data.shape[1],8),dtype=np.float32)
    dst = dev.createImage(data.shape[::-1],channel_type = cl.channel_type.FLOAT,mem_flags = cl.mem_flags.WRITE_ONLY)

    dev.writeImage(src,data.astype(np.float32))
    astride = roundDown(dst.width,8)

    for j in range(8):
            for i in range(8):
                proc.runKernel("dct_denoise_8x8_r",(roundDown(src.width-i,8),roundDown(src.height-j,8)),(8,8),
                               src,acc, np.float32(sigma),np.int32(i),np.int32(j),np.int32(astride),np.int32(i==0 and j==0))

    proc.runKernel("dct_denoise_normalise_r",(roundDown(src.width,8),roundDown(src.height,8)),(8,8),
                               acc, dst,np.int32(astride))


    out = dev.readBuffer(acc,dtype=np.float32)
    return np.maximum(0,dev.readImage(dst,dtype=np.float32)).astype(data.dtype)

def test_dct():
    from imgtools import test_images, calcPSNR

    data = test_images.lena()
    data = 100.*data/np.amax(data)
    y = data + np.random.normal(0,20,data.shape)
    y = y.astype(np.float32)

    out = dct_8x8(y,60)
    outs  = [calcPSNR(data,dct_8x8(y,s)) for s in np.linspace(10,60,20)]
    return out


def test_filter():
    from functools import partial
    import time

    data = np.random.uniform(0,1,[2**8,2**8]).astype(np.float32)

    print("running on image shape : {} \n\n".format(data.shape))

    fs = {
        "bilateral":lambda:bilateral(data,3,10,10),
        "nlMeans":lambda:nlm_fast(data,3,7,10),
        # "nlMeans S":lambda:nlMeansShared(clDev,data,3,7,10),
        # "nlMeans P":lambda:nlMeansProjected(clDev,data,3,7,10),
        "bm3d    ":lambda:bm3d(data,10),
        "dct    ":lambda:dct_denoising(data,10),
        "dct_8x8 ":lambda:dct_8x8(data,10),

    # "nlMeans prefetched":lambda:nlMeansPrefetch(clDev,data,3,5,10),
          #"nlMeans test":lambda:nlMeansTest(clDev,data,3,5,10),
          }

    t = time.time()
    for f in fs.keys:
        fs[f]()
        print("%s \t: \t %.2f s"%(f,time.time()-t))
        t = time.time()




if __name__ == '__main__':

    test_filter()


    # out1  = test_bilateral()

    # out  = test_nlm_fast()

    # out3  = test_nlm()

    # out4  = test_dct()



    #    test_filter()

    # from SpimUtils import test_images

    # data0 = test_images.lena()

    # y = np.maximum(0,1.*data0+np.random.normal(0,20,data0.shape)).astype(np.uint16)


    # dev = OCLDevice()

    # out = nlMeansProjected2(dev,y,3,4,35)
