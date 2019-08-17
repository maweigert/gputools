# sktensor.sptensor - base module for sparse tensors
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

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import issparse as issparse_mat
from sktensor.core import tensor_mixin
from sktensor.utils import accum
from sktensor.dtensor import unfolded_dtensor
from sktensor.pyutils import inherit_docstring_from, from_to_without


__all__ = [
    'concatenate',
    'fromarray',
    'sptensor',
    'unfolded_sptensor',
]


class sptensor(tensor_mixin):
    """
    A sparse tensor.

    Data is stored in COOrdinate format.

    Sparse tensors can be instantiated via

    Parameters
    ----------
    subs : n-tuple of array-likes
        Subscripts of the nonzero entries in the tensor.
        Length of tuple n must be equal to dimension of tensor.
    vals : array-like
        Values of the nonzero entries in the tensor.
    shape : n-tuple, optional
        Shape of the sparse tensor.
        Length of tuple n must be equal to dimension of tensor.
    dtype : dtype, optional
        Type of the entries in the tensor
    accumfun : function pointer
        Function to be accumulate duplicate entries

    Examples
    --------
    >>> S = sptensor(([0,1,2], [3,2,0], [2,2,2]), [1,1,1], shape=(10, 20, 5), dtype=np.float)
    >>> S.shape
    (10, 20, 5)
    >>> S.dtype
    <class 'float'>
    """

    def __init__(self, subs, vals, shape=None, dtype=None, accumfun=None, issorted=False):
        if not isinstance(subs, tuple):
            raise ValueError('Subscripts must be a tuple of array-likes')
        if len(subs[0]) != len(vals):
            raise ValueError('Subscripts and values must be of equal length')
        if dtype is None:
            dtype = np.array(vals).dtype
        for i in range(len(subs)):
            if np.array(subs[i]).dtype.kind != 'i':
                raise ValueError('Subscripts must be integers')

        vals = np.array(vals, dtype=dtype)
        if accumfun is not None:
            vals, subs = accum(
                subs, vals,
                issorted=False, with_subs=True, func=accumfun
            )
        self.subs = subs
        self.vals = vals
        self.dtype = dtype
        self.issorted = issorted
        self.accumfun = accumfun

        if shape is None:
            self.shape = tuple(np.array(subs).max(axis=1).flatten() + 1)
        else:
            self.shape = tuple(int(d) for d in shape)
        self.ndim = len(subs)

    def __eq__(self, other):
        if isinstance(other, sptensor):
            self._sort()
            other._sort()
            return (self.vals == other.vals).all() and (np.array(self.subs) == np.array(other.subs)).all()
        elif isinstance(other, np.ndarray):
            return (self.toarray() == other).all()
        else:
            raise NotImplementedError('Unsupported object class for sptensor.__eq__ (%s)' % type(other))

    def __getitem__(self, idx):
        # TODO check performance
        if len(idx) != self.ndim:
            raise ValueError('subscripts must be complete')
        sidx = np.ones(len(self.vals))
        for i in range(self.ndim):
            sidx = np.logical_and(self.subs[i] == idx[i], sidx)
        vals = self.vals[sidx]
        if len(vals) == 0:
            vals = 0
        elif len(vals) > 1:
            if self.accumfun is None:
                raise ValueError('Duplicate entries without specified accumulation function')
            vals = self.accumfun(vals)
        return vals

    def __sub__(self, other):
        if isinstance(other, np.ndarray):
            res = -other
            res[self.subs] += self.vals
        else:
            raise NotImplementedError()
        return res

    def _sort(self):
        # TODO check performance
        subs = np.array(self.subs)
        sidx = np.lexsort(subs)
        self.subs = tuple(z.flatten()[sidx] for z in np.vsplit(subs, len(self.shape)))
        self.vals = self.vals[sidx]
        self.issorted = True

    def _ttm_compute(self, V, mode, transp):
        Z = self.unfold(mode, transp=True).tocsr()
        if transp:
            V = V.T
        Z = Z.dot(V.T)
        shape = np.copy(self.shape)
        shape[mode] = V.shape[0]
        if issparse_mat(Z):
            newT = unfolded_sptensor((Z.data, (Z.row, Z.col)), [mode], None, shape=shape).fold()
        else:
            newT = unfolded_dtensor(Z.T, mode, shape).fold()

        return newT

    def _ttv_compute(self, v, dims, vidx, remdims):
        nvals = self.vals
        nsubs = self.subs
        for i in range(len(dims)):
            idx = nsubs[dims[i]]
            w = v[vidx[i]]
            nvals = nvals * w[idx]

        # Case 1: all dimensions used -> return sum
        if len(remdims) == 0:
            return nvals.sum()

        nsubs = tuple(self.subs[i] for i in remdims)
        nshp = tuple(self.shape[i] for i in remdims)

        # Case 2: result is a vector
        if len(remdims) == 1:
            usubs = np.unique(nsubs[0])
            bins = usubs.searchsorted(nsubs[0])
            c = np.bincount(bins, weights=nvals)
            (nz,) = c.nonzero()
            return sptensor((usubs[nz],), c[nz], nshp)

        # Case 3: result is an array
        return sptensor(nsubs, nvals, shape=nshp, accumfun=np.sum)

    def _ttm_me_compute(self, V, edims, sdims, transp):
        """
        Assume Y = T x_i V_i for i = 1...n can fit into memory
        """
        shapeY = np.copy(self.shape)

        # Determine size of Y
        for n in np.union1d(edims, sdims):
            shapeY[n] = V[n].shape[1] if transp else V[n].shape[0]

        # Allocate Y (final result) and v (vectors for elementwise computations)
        Y = np.zeros(shapeY)
        shapeY = np.array(shapeY)
        v = [None for _ in range(len(edims))]

        for i in range(np.prod(shapeY[edims])):
            rsubs = np.unravel_index(shapeY[edims], i)

    def unfold(self, rdims, cdims=None, transp=False):
        if isinstance(rdims, type(1)):
            rdims = [rdims]
        if transp:
            cdims = rdims
            rdims = np.setdiff1d(range(self.ndim), cdims)[::-1]
        elif cdims is None:
            cdims = np.setdiff1d(range(self.ndim), rdims)[::-1]
        if not (np.arange(self.ndim) == np.sort(np.hstack((rdims, cdims)))).all():
            raise ValueError(
                'Incorrect specification of dimensions (rdims: %s, cdims: %s)'
                % (str(rdims), str(cdims))
            )
        M = np.prod([self.shape[r] for r in rdims])
        N = np.prod([self.shape[c] for c in cdims])
        ridx = _build_idx(self.subs, self.vals, rdims, self.shape)
        cidx = _build_idx(self.subs, self.vals, cdims, self.shape)
        return unfolded_sptensor((self.vals, (ridx, cidx)), (M, N), rdims, cdims, self.shape)

    @inherit_docstring_from(tensor_mixin)
    def uttkrp(self, U, mode):
        R = U[1].shape[1] if mode == 0 else U[0].shape[1]
        dims = from_to_without(0, self.ndim, mode)
        V = np.zeros((self.shape[mode], R))
        for r in range(R):
            Z = tuple(U[n][:, r] for n in dims)
            TZ = self.ttv(Z, mode, without=True)
            if isinstance(TZ, sptensor):
                V[TZ.subs, r] = TZ.vals
            else:
                V[:, r] = self.ttv(Z, mode, without=True)
        return V

    @inherit_docstring_from(tensor_mixin)
    def transpose(self, axes=None):
        """
        Compute transpose of sparse tensors.

        Parameters
        ----------
        axes : array_like of ints, optional
            Permute the axes according to the values given.

        Returns
        -------
        d : dtensor
            dtensor with axes permuted.
        """
        if axes is None:
            raise NotImplementedError(
                'Sparse tensor transposition without axes argument is not supported'
            )
        nsubs = tuple([self.subs[idx] for idx in axes])
        nshape = [self.shape[idx] for idx in axes]
        return sptensor(nsubs, self.vals, nshape)

    def concatenate(self, tpl, axis=None):
        """
        Concatenates sparse tensors.

        Parameters
        ----------
        tpl :  tuple of sparse tensors
            Tensors to be concatenated.
        axis :  int, optional
            Axis along which concatenation should take place
        """
        if axis is None:
            raise NotImplementedError(
                'Sparse tensor concatenation without axis argument is not supported'
            )
        T = self
        for i in range(1, len(tpl)):
            T = _single_concatenate(T, tpl[i], axis=axis)
        return T

    def norm(self):
        """
        Frobenius norm for tensors

        References
        ----------
        [Kolda and Bader, 2009; p.457]
        """
        return np.linalg.norm(self.vals)

    def toarray(self):
        A = np.zeros(self.shape)
        A.put(np.ravel_multi_index(self.subs, tuple(self.shape)), self.vals)
        return A


class unfolded_sptensor(coo_matrix):
    """
    An unfolded sparse tensor.

    Data is stored in form of a sparse COO matrix.
    Unfolded_sptensor objects additionall hold information about the
    original tensor, such that re-folding the tensor into its original
    shape can be done easily.

    Unfolded_sptensor objects can be instantiated via

    Parameters
    ----------
    tpl : (data, (i, j)) tuple
        Construct sparse matrix from three arrays:
            1. ``data[:]``   the entries of the matrix, in any order
            2. ``i[:]``      the row indices of the matrix entries
            3. ``j[:]``      the column indices of the matrix entries
        where ``A[i[k], j[k]] = data[k]``.
    shape : tuple of integers
        Shape of the unfolded tensor.
    rdims : array_like of integers
        Modes of the original tensor that are mapped onto rows.
    cdims : array_like of integers
        Modes of the original tensor that are mapped onto columns.
    ten_shape : tuple of integers
        Shape of the original tensor.
    dtype : np.dtype, optional
        Data type of the unfolded tensor.
    copy : boolean, optional
        If true, data and subscripts are copied.

    Returns
    -------
    M : unfolded_sptensor
        Sparse matrix in COO format where ``rdims`` are mapped to rows and
        ``cdims`` are mapped to columns of the matrix.
    """

    def __init__(self, tpl, shape, rdims, cdims, ten_shape, dtype=None, copy=False):
        self.ten_shape = np.array(ten_shape)
        if isinstance(rdims, int):
            rdims = [rdims]
        if cdims is None:
            cdims = np.setdiff1d(range(len(self.ten_shape)), rdims)[::-1]
        self.rdims = rdims
        self.cdims = cdims
        super(unfolded_sptensor, self).__init__(tpl, shape=shape, dtype=dtype, copy=copy)

    def fold(self):
        """
        Recreate original tensor by folding unfolded_sptensor according toc
        ``ten_shape``.

        Returns
        -------
        T : sptensor
            Sparse tensor that is created by refolding according to ``ten_shape``.
        """
        nsubs = np.zeros((len(self.data), len(self.ten_shape)), dtype=np.int)
        if len(self.rdims) > 0:
            nidx = np.unravel_index(self.row, self.ten_shape[self.rdims])
            for i in range(len(self.rdims)):
                nsubs[:, self.rdims[i]] = nidx[i]
        if len(self.cdims) > 0:
            nidx = np.unravel_index(self.col, self.ten_shape[self.cdims])
            for i in range(len(self.cdims)):
                nsubs[:, self.cdims[i]] = nidx[i]
        nsubs = [z.flatten() for z in np.hsplit(nsubs, len(self.ten_shape))]
        return sptensor(tuple(nsubs), self.data, self.ten_shape)


def fromarray(A):
    """Create a sptensor from a dense numpy array"""
    subs = np.nonzero(A)
    vals = A[subs]
    return sptensor(subs, vals, shape=A.shape, dtype=A.dtype)


def _single_concatenate(ten, other, axis):
    tshape = ten.shape
    oshape = other.shape
    if len(tshape) != len(oshape):
        raise ValueError("len(tshape) != len(oshape")
    oaxes = np.setdiff1d(range(len(tshape)), [axis])
    for i in oaxes:
        if tshape[i] != oshape[i]:
            raise ValueError("Dimensions must match")
    nsubs = [None for _ in range(len(tshape))]
    for i in oaxes:
        nsubs[i] = np.concatenate((ten.subs[i], other.subs[i]))
    nsubs[axis] = np.concatenate((
        ten.subs[axis], other.subs[axis] + tshape[axis]
    ))
    nvals = np.concatenate((ten.vals, other.vals))
    nshape = np.copy(tshape)
    nshape[axis] = tshape[axis] + oshape[axis]
    return sptensor(nsubs, nvals, nshape)


def _build_idx(subs, vals, dims, tshape):
    shape = np.array([tshape[d] for d in dims], ndmin=1)
    dims = np.array(dims, ndmin=1)
    if len(shape) == 0:
        idx = np.ones(len(vals), dtype=vals.dtype)
    elif len(subs) == 0:
        idx = np.array(tuple())
    else:
        idx = np.ravel_multi_index(tuple(subs[i] for i in dims), shape)
    return idx
