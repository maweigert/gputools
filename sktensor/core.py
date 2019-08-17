# Copyright (C) 2013 Maximilian Nickel <mnick@mit.edu>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import sys
import types
import abc
from collections.abc import Sequence
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import issparse as issparse_mat
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

module_funs = []


def modulefunction(func):
    module_funs.append(func.__name__)


class tensor_mixin:
    """
    Base tensor class from which all tensor classes are subclasses.
    Can not be instantiated

    See also
    --------
    sktensor.dtensor : Subclass for *dense* tensors.
    sktensor.sptensor : Subclass for *sparse* tensors.
    """

    __metaclass__ = abc.ABCMeta

    def ttm(self, V, mode=None, transp=False, without=False):
        """
        Tensor times matrix product

        Parameters
        ----------
        V : M x N array_like or list of M_i x N_i array_likes
            Matrix or list of matrices for which the tensor times matrix
            products should be performed
        mode : int or list of int's, optional
            Modes along which the tensor times matrix products should be
            performed
        transp: boolean, optional
            If True, tensor times matrix products are computed with
            transpositions of matrices
        without: boolean, optional
            It True, tensor times matrix products are performed along all
            modes **except** the modes specified via parameter ``mode``


        Examples
        --------
        Create dense tensor

        >>> from sktensor import dtensor
        >>> T = np.zeros((3, 4, 2))
        >>> T[:, :, 0] = [[ 1,  4,  7, 10], [ 2,  5,  8, 11], [3,  6,  9, 12]]
        >>> T[:, :, 1] = [[13, 16, 19, 22], [14, 17, 20, 23], [15, 18, 21, 24]]
        >>> T = dtensor(T)

        Create matrix

        >>> V = np.array([[1, 3, 5], [2, 4, 6]])

        Multiply tensor with matrix along mode 0

        >>> Y = T.ttm(V, 0)
        >>> Y[:, :, 0]
        dtensor([[ 22.,  49.,  76., 103.],
                 [ 28.,  64., 100., 136.]])
        >>> Y[:, :, 1]
        dtensor([[130., 157., 184., 211.],
                 [172., 208., 244., 280.]])

        """
        if mode is None:
            mode = range(self.ndim)
        if isinstance(V, np.ndarray):
            Y = self._ttm_compute(V, mode, transp)
        elif isinstance(V, Sequence):
            dims, vidx = check_multiplication_dims(mode, self.ndim, len(V), vidx=True, without=without)
            Y = self._ttm_compute(V[vidx[0]], dims[0], transp)
            for i in range(1, len(dims)):
                Y = Y._ttm_compute(V[vidx[i]], dims[i], transp)
        return Y

    def ttv(self, v, modes=None, without=False):
        """
        Tensor times vector product

        Parameters
        ----------
        v : 1-d array or tuple of 1-d arrays
            Vector to be multiplied with tensor.
        modes : array_like of integers, optional
            Modes in which the vectors should be multiplied.
        without : boolean, optional
            If True, vectors are multiplied in all modes **except** the
            modes specified in ``modes``.

        """
        if modes is None:
            modes = []
        if not isinstance(v, tuple):
            v = (v, )
        dims, vidx = check_multiplication_dims(modes, self.ndim, len(v), vidx=True, without=without)
        for i in range(len(dims)):
            if not len(v[vidx[i]]) == self.shape[dims[i]]:
                raise ValueError('Multiplicant is wrong size')
        remdims = np.setdiff1d(range(self.ndim), dims)
        return self._ttv_compute(v, dims, vidx, remdims)

    @abc.abstractmethod
    def _ttm_compute(self, V, mode, transp):
        pass

    @abc.abstractmethod
    def _ttv_compute(self, v, dims, vidx, remdims):
        pass

    @abc.abstractmethod
    def unfold(self, rdims, cdims=None, transp=False):
        pass

    @abc.abstractmethod
    def uttkrp(self, U, mode):
        """
        Unfolded tensor times Khatri-Rao product:
        :math:`M = (U_1 \\otimes \\cdots \\otimes U_N)`

        Computes the _matrix_ product of the unfolding
        of a tensor and the Khatri-Rao product of multiple matrices.
        Efficient computations are perfomed by the respective
        tensor implementations.

        Parameters
        ----------
        U : list of array-likes
            Matrices for which the Khatri-Rao product is computed and
            which are multiplied with the tensor in mode ``mode``.
        mode: int
            Mode in which the Khatri-Rao product of ``U`` is multiplied
            with the tensor.

        Returns
        -------
        M : np.ndarray
            Matrix which is the result of the matrix product of the unfolding of
            the tensor and the Khatri-Rao product of ``U``

        See Also
        --------
        dtensor.uttkrp, sptensor.uttkrp, ktensor.uttkrp, ttensor.uttkrp : For
            efficient computations of unfolded tensor times Khatri-Rao products
            for specialiized tensors.

        References
        ----------
        .. [1] B.W. Bader, T.G. Kolda
               Efficient Matlab Computations With Sparse and Factored Tensors
               SIAM J. Sci. Comput, Vol 30, No. 1, pp. 205--231, 2007

        """
        pass

    @abc.abstractmethod
    def transpose(self, axes=None):
        """
        Compute transpose of tensors.

        Parameters
        ----------
        axes : array_like of ints, optional
            Permute the axes according to the values given.

        Returns
        -------
        d : tensor_mixin
            tensor with axes permuted.

        See also
        --------
        dtensor.transpose, sptensor.transpose
        """
        pass


def istensor(X):
    return isinstance(X, tensor_mixin)


# dynamically create module level functions
conv_funcs = [
    'norm',
    'transpose',
    'ttm',
    'ttv',
    'unfold',
]

for fname in conv_funcs:
    def call_on_me(obj, *args, **kwargs):
        if not istensor(obj):
            raise ValueError('%s() object must be tensor (%s)' % (fname, type(obj)))
        func = getattr(obj, fname)
        return func(*args, **kwargs)

    nfunc = types.FunctionType(
        call_on_me.__code__,
        {
            'getattr': getattr,
            'fname': fname,
            'istensor': istensor,
            'ValueError': ValueError,
            'type': type
        },
        name=fname,
        argdefs=call_on_me.__defaults__,
        closure=call_on_me.__closure__
    )
    setattr(sys.modules[__name__], fname, nfunc)


def check_multiplication_dims(dims, N, M, vidx=False, without=False):
    dims = np.array(dims, ndmin=1)
    if len(dims) == 0:
        dims = np.arange(N)
    if without:
        dims = np.setdiff1d(range(N), dims)
    if not np.in1d(dims, np.arange(N)).all():
        raise ValueError('Invalid dimensions')
    P = len(dims)
    sidx = np.argsort(dims)
    sdims = dims[sidx]
    if vidx:
        if M > N:
            raise ValueError('More multiplicants than dimensions')
        if M != N and M != P:
            raise ValueError('Invalid number of multiplicants')
        if P == M:
            vidx = sidx
        else:
            vidx = sdims
        return sdims, vidx
    else:
        return sdims


def innerprod(X, Y):
    """
    Inner prodcut with a Tensor
    """
    return np.dot(X.flatten(), Y.flatten())


def nvecs(X, n, rank, do_flipsign=True, dtype=np.float):
    """
    Eigendecomposition of mode-n unfolding of a tensor
    """
    Xn = X.unfold(n)
    if issparse_mat(Xn):
        Xn = csr_matrix(Xn, dtype=dtype)
        Y = Xn.dot(Xn.T)
        _, U = eigsh(Y, rank, which='LM')
    else:
        Y = Xn.dot(Xn.T)
        N = Y.shape[0]
        _, U = eigh(Y, eigvals=(N - rank, N - 1))
    # reverse order of eigenvectors such that eigenvalues are decreasing
    U = np.array(U[:, ::-1])
    # flip sign
    if do_flipsign:
        U = flipsign(U)
    return U


def flipsign(U):
    """
    Flip sign of factor matrices such that largest magnitude
    element will be positive
    """
    midx = abs(U).argmax(axis=0)
    for i in range(U.shape[1]):
        if U[midx[i], i] < 0:
            U[:, i] = -U[:, i]
    return U


def center(X, n):
    Xn = unfold(X, n)
    N = Xn.shape[0]
    m = Xn.sum(axis=0) / N
    m = np.kron(m, np.ones((N, 1)))
    Xn = Xn - m
    return fold(Xn, n)


def center_matrix(X):
    m = X.mean(axis=0)
    return X - m


def scale(X, n):
    Xn = unfold(X, n)
    m = np.float_(np.sqrt((Xn ** 2).sum(axis=1)))
    m[m == 0] = 1
    for i in range(Xn.shape[0]):
        Xn[i, :] = Xn[i] / m[i]
    return fold(Xn, n, X.shape)


# TODO more efficient cython implementation
def khatrirao(A, reverse=False):
    """
    Compute the columnwise Khatri-Rao product.

    Parameters
    ----------
    A : tuple of ndarrays
        Matrices for which the columnwise Khatri-Rao product should be computed

    reverse : boolean
        Compute Khatri-Rao product in reverse order

    Examples
    --------
    >>> A = np.random.randn(5, 2)
    >>> B = np.random.randn(4, 2)
    >>> C = khatrirao((A, B))
    >>> C.shape
    (20, 2)
    >>> (C[:, 0] == np.kron(A[:, 0], B[:, 0])).all()
    True
    >>> (C[:, 1] == np.kron(A[:, 1], B[:, 1])).all()
    True
    """

    if not isinstance(A, tuple):
        raise ValueError('A must be a tuple of array likes')
    N = A[0].shape[1]
    M = 1
    for i in range(len(A)):
        if A[i].ndim != 2:
            raise ValueError('A must be a tuple of matrices (A[%d].ndim = %d)' % (i, A[i].ndim))
        elif N != A[i].shape[1]:
            raise ValueError('All matrices must have same number of columns')
        M *= A[i].shape[0]
    matorder = np.arange(len(A))
    if reverse:
        matorder = matorder[::-1]
    # preallocate
    P = np.zeros((M, N), dtype=A[0].dtype)
    for n in range(N):
        ab = A[matorder[0]][:, n]
        for j in range(1, len(matorder)):
            ab = np.kron(ab, A[matorder[j]][:, n])
        P[:, n] = ab
    return P


def tvecmat(m, n):
    d = m * n
    i2 = np.arange(d).reshape(m, n).T.flatten()
    Tmn = np.zeros((d, d))
    Tmn[np.arange(d), i2] = 1
    return Tmn

# vim: set et:
