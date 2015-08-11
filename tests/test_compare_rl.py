import numpy as np

from scipy.signal import convolve2d


def richardson_lucy(image, psf, iterations=50, clip=True):
    image = image.astype(np.float)
    psf = psf.astype(np.float)
    im_deconv = 0.5 * np.ones(image.shape)
    psf_mirror = psf[::-1, ::-1]
    for _ in range(iterations):
        relative_blur = image / convolve2d(im_deconv, psf, 'same')
        im_deconv *= convolve2d(relative_blur, psf_mirror, 'same')
        # return relative_blur
    
    if clip:
        im_deconv[im_deconv > 1] = 1
        im_deconv[im_deconv < -1] = -1

    return im_deconv

                
from skimage import color, data, restoration
camera = color.rgb2gray(data.camera())
psf = np.ones((21, 21)) / 21**2
camera = convolve2d(camera, psf, 'same')
camera += 0.01 * camera.std() * np.random.standard_normal(camera.shape)
out = restoration.richardson_lucy(camera, psf, 15, clip = False)
