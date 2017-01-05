from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np


from gputools import OCLArray, OCLImage

def image_from_array(data):
    im = OCLImage.from_array(data)
    assert np.allclose(data,im.get())


def image_create_write(data):
    im = OCLImage.empty(data.shape,data.dtype)
    im.write_array(data)
    
    assert np.allclose(data,im.get())

def buffer_from_array(data):
    buf = OCLArray.from_array(data)
    assert np.allclose(data,buf.get())

def buffer_create_write(data):
    buf = OCLArray.empty(data.shape,data.dtype)
    buf.write_array(data)
    assert np.allclose(data,buf.get())


def test_all():
    ndims = [2,3]
    Ns = [10,100,200]

    for N in Ns:
        for ndim in ndims:
            shape = [N+n for n in ndims[:ndim]]
            print("testing creation and writing %s"%(shape))
            data = np.linspace(0,1,np.prod(shape)).reshape(shape).astype(np.float32)
            image_create_write(data)
            image_from_array(data)
            buffer_create_write(data)
            buffer_from_array(data)

    
if __name__ == '__main__':

    test_all()
