from gputools import fft, fft_convolve, fft_plan

import numpy as np

def test_fft_np():
    d = np.ones((128,)*2)
    res = fft(d)


def test_fftconv_np():
    d = np.ones((128,)*2, np.float32)
    res = fft_convolve(d,d)
    
if __name__ == '__main__':
    pass
    



