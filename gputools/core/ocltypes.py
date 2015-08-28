'''
@author: mweigert

A basic wrapper class around pyopencl.cl__array

'''
import numpy as np
import pyopencl.array as cl_array
import pyopencl

from gputools import get_device

import pyopencl.clmath as cl_math
import collections


def assert_bufs_type(mytype,*bufs):
    if not all([b.dtype.type == mytype for b in bufs]):
        raise TypeError("all data type of buffer(s) should be %s! but are %s"%
                        (mytype,str([b.dtype.type for b in bufs])))


def _wrap_OCLArray(cls):
    """
    WRAPPER
    """
    @classmethod
    def from_array(cls,arr,*args, **kwargs):
        queue = get_device().queue
        return cl_array.to_device(queue, arr,*args, **kwargs)
    @classmethod
    def empty(cls, shape, dtype = np.float32):
        queue = get_device().queue
        return cl_array.empty(queue, shape,dtype)

    @classmethod
    def empty_like(cls, arr):
        return cls.empty(arr.shape,arr.dtype)

    @classmethod
    def zeros(cls, shape, dtype = np.float32):
        queue = get_device().queue
        return cl_array.zeros(queue, shape,dtype)

    @classmethod
    def zeros_like(cls, arr):
        return cl_array.zeros_like(queue, arr)

    def copy_buffer(self,buf, **kwargs):
        queue = get_device().queue
        return pyopencl.enqueue_copy(queue, self.data, buf.data,
                                      **kwargs)

    def write_array(self,data, **kwargs):
        queue = get_device().queue
        return pyopencl.enqueue_write_buffer(queue, self.data, data,
                                      **kwargs)
    
    def copy_image(self,img, **kwargs):
        queue = get_device().queue
        return pyopencl.enqueue_copy(queue, self.data, img, offset = 0,
                                     origin = (0,0),region = img.shape,
                                      **kwargs)

    def wrap_module_func(mod,f):
        def func(self,*args,**kwargs):
            return getattr(mod,f)(self,*args,**kwargs)
        return func

    
    cls.from_array = from_array
    cls.empty = empty
    cls.empty_like = empty_like
    cls.zeros = zeros
    cls.zeros_like = zeros_like

    cls.copy_buffer = copy_buffer
    cls.copy_image = copy_image
    cls.write_array = write_array

    for f in ["sum","max","min","dot","vdot"]:
        setattr(cls,f,wrap_module_func(cl_array,f))

    for f in dir(cl_math):
        if isinstance(getattr(cl_math,f), collections.Callable):
            setattr(cls,f,wrap_module_func(cl_math,f))

    
    # cls.sum = sum
    cls.__name__ = "OCLArray"
    return cls

def _wrap_OCLImage(cls):
    @classmethod
    def from_array(cls,arr, *args, **kwargs):
        ctx = get_device().context
        if not arr.ndim in [1,2,3,4]:
            raise ValueError("dimension of array wrong, should be 1...4 but is %s"%arr.ndim)        
        elif arr.ndim == 4:
            num_channels = arr.shape[-1]
        else:
            num_channels = None
        
        res =  pyopencl.image_from_array(ctx, arr,num_channels = num_channels,
                                             *args, **kwargs)
        res.dtype = arr.dtype
        return res

    @classmethod
    def zeros(cls, shape, dtype = np.float32):
        queue = get_device().queue
        return cl_array.zeros(queue, shape,dtype)
    
    @classmethod
    def empty(cls,shape,dtype, num_channels = 1, channel_order = None):
        ctx = get_device().context
        if not len(shape) in [1,2,3]:
            raise ValueError("dimension of shape wrong, should be 1...3 but is %s"%len(shape))
        mem_flags = pyopencl.mem_flags.READ_WRITE
        channel_type = pyopencl.DTYPE_TO_CHANNEL_TYPE[np.dtype(dtype)]


        _dict_channel_order = {1:pyopencl.channel_order.R,
                               2:pyopencl.channel_order.RG,
                               3:pyopencl.channel_order.RGB,
                               4:pyopencl.channel_order.RGBA}

        if channel_order is None:
            channel_order = _dict_channel_order[num_channels]
            
        fmt = pyopencl.ImageFormat(channel_order, channel_type)
            
        res =  pyopencl.Image(ctx, mem_flags,fmt, shape = shape[::-1])            
        res.dtype = dtype
        return res

    #@classmethod
    # def empty(cls,shape,dtype):
    #     ctx = get_device().context
    #     if not len(shape) in [1,2,3,4]:
    #         raise ValueError("dimension of shape wrong, should be 1...4 but is %s"%len(shape))
    #     elif len(shape) == 4:
    #         num_channels = shape[-1]
    #         channel_order = pyopencl.channel_order.RGBA
    #         shape = shape[:-1]

    #     else:
    #         num_channels = None
    #         channel_order = pyopencl.channel_order.R

    #     mem_flags = pyopencl.mem_flags.READ_WRITE
    #     channel_type = pyopencl.DTYPE_TO_CHANNEL_TYPE[dtype]
            
    #     fmt = pyopencl.ImageFormat(channel_order, channel_type)
            
    #     res =  pyopencl.Image(ctx, mem_flags,fmt, shape = shape[::-1])            
    #     res.dtype = dtype
    #     return res
       
    @classmethod
    def empty_like(cls,arr):
        return cls.empty(arr.shape,arr.dtype)
        
       
    def copy_buffer(self,buf, **kwargs):
        queue = get_device().queue
        self.dtype = buf.dtype
        return pyopencl.enqueue_copy(queue, self, buf.data, offset=0,
                                     origin = (0,0),region = self.shape,  **kwargs)

    def copy_image(self,img, **kwargs):
        queue = get_device().queue
        return pyopencl.enqueue_copy(queue, self, img, 
                                     src_origin = (0,0),
                                     dest_origin = (0,0),
                                     region = img.shape,
                                      **kwargs)

    def write_array(self, data):
        queue = get_device().queue        

        # 1d images dont have a shape but only a width
        if hasattr(self,"shape"):
            imshape = self.shape
        else:
            imshape = (self.width,)

        ndim = len(imshape)
        dshape = data.shape
        # if clImg.format.channel_order in [cl.channel_order.RGBA,
        #                                   cl.channel_order.BGRA]:
        #     dshape = dshape[:-1]

        if dshape != imshape[::-1]:
            raise ValueError("write_array: wrong shape!",data.shape[::-1],imshape)
        else:
            pyopencl.enqueue_write_image(queue,self,[0]*ndim,imshape,data)

    def get(self, **kwargs):
        queue = get_device().queue
        if hasattr(self,"shape"):
            imshape = self.shape
        else:
            imshape = (self.width,)

        dshape = imshape[::-1]
        if self.format.channel_count>1:
            dshape += (self.format.channel_count,)
        out = np.empty(dshape,dtype=self.dtype)
        pyopencl.enqueue_read_image(queue,self,[0]*len(dshape),imshape,out)

        return out.reshape(dshape)
    
    cls.from_array = from_array
    cls.empty = empty
    cls.empty_like = empty_like

    cls.copy_buffer = copy_buffer
    cls.copy_image = copy_image
    cls.write_array = write_array

    cls.get = get

    cls.__name__ = "OCLImage"
    return cls


OCLArray = _wrap_OCLArray(cl_array.Array)
OCLImage = _wrap_OCLImage(pyopencl.Image)


def test_types():
    d = np.random.uniform(0,1,(40,50,60)).astype(np.float32)

    b0 = OCLArray.from_array(d)

    im0 = OCLImage.from_array(d)

    b1 = OCLArray.empty_like(d)
    b2 = OCLArray.empty_like(d)

    im1 = OCLImage.empty_like(d)
    im2 = OCLImage.empty_like(d)

    b1.copy_buffer(b0)
    b2.copy_image(im0)

    im1.copy_buffer(b0)
    im2.copy_image(im0)

    for x in [b0,b1,b2,im0,im1,im2]:
        if hasattr(x,"sum"):
            print(("sum: %s" %x.sum()))
        assert np.allclose(d,x.get())

        
if __name__ == '__main__':

    test_types()


    d = np.linspace(0,1,100).astype(np.float32)

    b = OCLArray.from_array(d)
