import numpy as np
from sktensor import tucker
from sktensor.core import ttm
from sktensor.dtensor import dtensor, unfolded_dtensor
from sktensor.sptensor import unfolded_sptensor


def normalize(X):
   return X / X.sum(axis=0)


def disabled_test_factorization():
    I, J, K, rank = 10, 20, 75, 5
    A = orthomax(np.random.randn(I, rank))
    B = orthomax(np.random.randn(J, rank))
    C = orthomax(np.random.randn(K, rank))

    core_real = dtensor(np.random.randn(rank, rank, rank))
    T = core_real.ttm([A, B, C])
    core, U = tucker.hooi(T, rank)

    assert np.allclose(T, ttm(core, U))
    assert np.allclose(A, orthomax(U[0]))
    assert np.allclose(B, orthomax(U[1]))
    assert np.allclose(C, orthomax(U[2]))
    assert np.allclose(core_real, core)


def disabled_test_factorization_sparse():
    I, J, K, rank = 10, 20, 75, 5
    Tmat = scipy.spare.rand(I, J * K, 0.1).tocoo()
    T = unfolded_sptensor((Tmat.data, (Tmat.row, Tmat.col)), None, 0, [], (I, J, K)).fold()
    core, U = tucker.hooi(T, rank, maxIter=20)

    Tmat = Tmat.toarray()
    T = unfolded_dtensor(Tmat, 0, (I, J, K)).fold()
    core2, U2 = tucker.hooi(T, rank, maxIter=20)

    assert np.allclose(core2, core)
    for i in range(len(U)):
        assert np.allclose(U2[i], U[i])
