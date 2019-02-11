import numpy as np
from skimage.transform import integral_image as sk_integral_image
from gputools.transforms import integral_image
from gputools import get_device
from itertools import product, combinations, permutations

type_name_dict = {
    np.uint8: "uint8",
    np.uint16: "uint16",
    np.float32: "float32",
    np.complex64: "complex64",
}



def single_test(shape, dtype = np.float32, check = True):
    np.random.seed(42)
    x = np.random.randint(0, 243, shape).astype(dtype)
    y1 = sk_integral_image(x)
    y2 = integral_image(x)
    if check:
        is_close = np.allclose(y1, y2)
        print("integral_image: %s %s\t%s" % (str(shape), type_name_dict[dtype], is_close))
        assert is_close
    return y1,y2

def test_integral():
    max_size = get_device().get_info("GLOBAL_MEM_SIZE") // 32
    ndims = (2,3)
    ns = (33, 197, 4197)
    dtypes = (np.uint8, np.uint16, np.int32, np.float64)
    for dtype, ndim in product(dtypes, ndims):
        for shape in combinations(ns, ndim):
            if np.prod(shape) > max_size:
                continue
            single_test(shape, check = True)

if __name__ == '__main__':
    # test_integral()

    y1, y2 = single_test((4,256), np.int32, check=False)
    #y1, y2 = single_test((11,11,11), np.uint8, check=False)
