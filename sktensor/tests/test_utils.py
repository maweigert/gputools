import numpy as np
from ..utils import accum


def test_accum():
    subs1 = np.array([0, 1, 1, 2, 2, 2])
    subs2 = np.array([0, 1, 1, 1, 2, 2])
    vals = np.array([1, 2, 3, 4, 5, 6])
    nvals, nsubs = accum((subs1, subs2), vals, with_subs=True)
    assert np.allclose(nvals, np.array([1, 5, 4, 11]))
    assert np.allclose(nsubs[0], np.array([0, 1, 2, 2]))
    assert np.allclose(nsubs[1], np.array([0, 1, 1, 2]))

    subs1 = np.array([0, 0, 1])
    subs2 = np.array([0, 0, 1])
    vals = np.array([1, 2, 3])
    nvals, nsubs = accum((subs1, subs2), vals, with_subs=True)
    assert np.allclose(nvals, np.array([3, 3]))
    assert np.allclose(nsubs[0], np.array([0, 1]))
    assert np.allclose(nsubs[1], np.array([0, 1]))
