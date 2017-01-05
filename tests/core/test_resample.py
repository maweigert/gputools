"""test transfers from buffer to image and so on..."""

from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from gputools import OCLArray, OCLImage


def transfer(data):
    """transfers data"""

    d1_g = OCLArray.from_array(data)
    d2_g = OCLArray.empty_like(data)

    if data.dtype.type == np.float32:
        im = OCLImage.empty(data.shape[::1],dtype = np.float32)
    elif data.dtype.type == np.complex64:
        im = OCLImage.empty(data.shape[::1],dtype = np.float32, num_channels=2)

    im.copy_buffer(d1_g)
    d2_g.copy_image(im)

    return d2_g.get()

def resample_img(data, new_shape):
    """resamples d"""

    d1_g = OCLImage.from_array(data)
    d2_g = OCLImage.empty(new_shape,np.float32,num_channels = 2 if np.iscomplexobj(data) else 1)

    d2_g.copy_image_resampled(d1_g)

    return d2_g.get()

def resample_buf(data, new_shape):
    """resamples d"""

    d1_g = OCLArray.from_array(data)
    d2_g = OCLArray.empty(new_shape,data.dtype)

    if data.dtype.type == np.float32:
        im = OCLImage.empty(data.shape[::1],dtype = np.float32)
    elif data.dtype.type == np.complex64:
        im = OCLImage.empty(data.shape[::1],dtype = np.float32, num_channels=2)

    im.copy_buffer(d1_g)
    d2_g.copy_image_resampled(im)

    return d2_g.get()

def test_transfer():

    d1 = np.random.uniform(-1,1,(256,400)).astype(np.float32)
    d2 = transfer(d1)
    assert np.allclose(d1,d2)

    d1 = np.random.uniform(-1,1,(101,103,107)).astype(np.float32)
    d2 = transfer(d1)
    assert np.allclose(d1,d2)

    d1 = np.random.uniform(-1,1,(256,400)).astype(np.complex64)
    d2 = transfer(d1)
    assert np.allclose(d1,d2)

    d1 = np.random.uniform(-1,1,(101,103,107)).astype(np.complex64)
    d2 = transfer(d1)
    assert np.allclose(d1,d2)

if __name__ == '__main__':

    #test_transfer()

    d1 = np.random.uniform(-1,1,(200,400)).astype(np.float32)
    d1[10:40,10:60] += 100
    d2 = resample_buf(d1, (300, 600))


    d1 = np.random.uniform(-1,1,(101,103,197)).astype(np.float32)
    d1[10:40,10:60,40:100] += 100
    d2 = resample_buf(d1, (201, 205, 307))

    d1 = np.random.uniform(-1,1,(200,400)).astype(np.complex64)
    d1 += 1.j*np.random.uniform(-1,1,(200,400)).astype(np.complex64)
    d1[10:40,10:60] += 10

    d2 = resample_buf(d1, (300, 600))

    im2 = resample_img(d1, (100, 200))
