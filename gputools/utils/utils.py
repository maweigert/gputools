import numpy as np

def pad_to_shape(d , dshape, mode = "constant"):
    if d.shape == dshape:
        return d

    diff = np.array(dshape)- np.array(d.shape)
    #first shrink
    slices  = [slice(-x/2,x/2) if x<0 else slice(None,None) for x in diff]
    res = d[slices]
    #then padd
    return np.pad(res,[(n/2,n-n/2) if n>0 else (0,0) for n in diff],mode=mode)


def _is_power2(n):
    return _next_power_of_2(n) == n

def _next_power_of_2(n):
    return int(2**np.ceil(np.log2(n)))

def pad_to_power2(data, mode="constant"):
    if np.all([_is_power2(n) for n in data.shape]):
        return data
    else:
        return pad_to_shape(data,[_next_power_of_2(n) for n in data.shape],mode)


def get_cache_dir():
    from tempfile import gettempdir
    import getpass
    import os
    import sys

    return os.path.join(gettempdir(),
                    "pyopencl-compiler-cache-v2-uid%s-py%s" % (
                        getpass.getuser(), ".".join(str(i) for i in sys.version_info)))

def remove_cache_dir():
    import shutil

    cache_dir = get_cache_dir()
    print("try removing cache dir: %s"%cache_dir)
    try:
        shutil.rmtree(cache_dir)
    except Exception as e:
        print(e)



