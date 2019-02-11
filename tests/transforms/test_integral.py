import numpy as np
from skimage.transform import integral_image as sk_integral_image
from gputools.transforms import integral_image
from gputools import get_device
from itertools import product


def test_integral():
    max_size = get_device().get_info("GLOBAL_MEM_SIZE") // 32

    ndims = (2, 3)
    ns = (11, 128, 197, 4197)
    for ndim in ndims:
        for shape in product(ns, repeat=ndim):
            if np.prod(shape) > max_size:
                continue
            x = np.random.randint(-100, 100, shape).astype(np.float32)
            y1 = sk_integral_image(x)
            y2 = integral_image(x)
            check = np.allclose(y1, y2)
            print("integral_image: %s\t%s" % (str(shape), check))
            assert check


if __name__ == '__main__':
    test_integral()
