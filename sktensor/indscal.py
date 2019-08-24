import time
import logging
import numpy as np
from scipy.linalg import svd, norm, orth
from scipy.sparse.linalg import eigsh

_log = logging.getLogger('INDSCAL')

_DEF_MAXITER = 50
_DEF_INIT = 'random'
_DEF_CONV = 1e-7


def orth_als(X, ncomp, **kwargs):

    ainit = kwargs.pop('init', _DEF_INIT)
    maxiter = kwargs.pop('max_iter', _DEF_MAXITER)
    conv = kwargs.pop('conv', _DEF_CONV)
    if not len(kwargs) == 0:
        raise ValueError('Unknown keywords (%s)' % (kwargs.keys()))

    K = len(X)
    normX = sum([norm(Xk)**2 for Xk in X])

    A = init(X, ainit, ncomp)
    fit = 0
    exectimes = []
    for itr in range(maxiter):
        tic = time.time()
        fitold = fit
        D = _updateD(X, A)
        A = _updateA(X, A, D)

        fit = sum([norm(X[k] - np.dot(A, np.dot(np.diag(D[k, :]), A.T)))**2 for k in range(K)])
        fit = 1 - fit / normX
        fitchange = abs(fitold - fit)

        exectimes.append(time.time() - tic)
        _log.info('[%3d] fit: %0.5f | delta: %7.1e | secs: %.5f' % (
            itr, fit, fitchange, exectimes[-1]
        ))
        if itr > 0 and fitchange < conv:
            break
    return A, D


def _updateA(X, A, D):
    G = np.zeros(A.shape)
    for k in range(len(X)):
        G = G + np.dot(X[k], np.dot(A, np.diag(D[k, :])))
    U, _, Vt = svd(G, full_matrices=0)
    A = np.dot(U, Vt)
    return A


def _updateD(X, A):
    K, R = len(X), A.shape[1]
    D = np.zeros((K, R))
    for k in range(K):
        D[k, :] = np.diag(np.dot(A.T, np.dot(X[k], A)))
    D[D < 0] = 0
    return D


def init(X, init, ncomp):
    N, K = X[0].shape[0], len(X)
    if init == 'random':
        A = orth(np.random.rand(N, ncomp))
    elif init == 'nvecs':
        S = np.zeros(N, N)
        for k in range(K):
            S = S + X[k] + X[k].T
        _, A = eigsh(S, ncomp)
    return A
