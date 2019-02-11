import numpy as np
from gputools.utils.histogram import histogram
from time import time
import pytest

@pytest.mark.skip(reason="WIP")
def test_histograms(return_if_fail=False):
    np.random.seed(0)

    for n in 2 ** np.arange(4, 22):
        dtype = np.random.choice((np.uint8, np.uint16, np.float32))
        x = np.random.randint(0, np.random.randint(10, 200), n).astype(dtype)
        n_bins = np.random.randint(10, 400)

        h1 = np.histogram(x, n_bins)[0]
        h2 = histogram(x, n_bins)

        equal = np.allclose(h1, h2)
        print(dtype, equal)
        if not equal and return_if_fail:
            return x, n_bins, h1, h2,
        else:
            assert equal


if __name__ == '__main__':
    np.random.seed(0)
    x = np.random.randint(0,256,4096**2).astype(np.uint16)
    # x = np.random.randint(0, 256, 1024).astype(np.uint16)

    n_bins = 256
    t = time()
    h1 = np.histogram(x, n_bins)[0]
    print("time cpu: %.4f ms"%(1000*(time()-t)))

    t = time()
    h2 = histogram(x, n_bins)
    print("time gpu: %.4f ms" % (1000 * (time() - t)))
