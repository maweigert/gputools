from __future__ import print_function, division

import numpy as np
from gputools.utils import tile_iterator

if __name__ == '__main__':

    x = np.random.uniform(-1, 1, (100, 100, 100))

    y = np.empty_like(x)

    for x_tile, src, dest in tile_iterator(x, (64,) * x.ndim, padsize=(10,) * x.ndim):
        print(x_tile.shape, src, dest)
        y[src] = x_tile[dest]


    print(np.allclose(x,y))
    assert np.allclose(x,y)