import numpy as np
import pytest


@pytest.fixture
def sptensor_seed():
    return np.random.seed(5)


@pytest.fixture
def sz():
    return 100


@pytest.fixture
def vals(sptensor_seed, sz):
    return np.random.randint(0, 100, sz)


@pytest.fixture
def shape():
    return (25, 11, 18, 7, 2)


@pytest.fixture
def subs(sptensor_seed, shape, sz):
    return tuple(np.random.randint(0, shape[i], sz) for i in range(len(shape)))
