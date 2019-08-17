# sktensor.dtensor - base class for dense tensors
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
from .core import tensor_mixin, khatrirao
from .pyutils import inherit_docstring_from, from_to_without


__all__ = [
    'dtensor',
    'unfolded_dtensor',
]


class dtensor(tensor_mixin, np.ndarray):
    """
    Class to store **dense** tensors

    Parameters
    ----------
    input_array : np.ndarray
        Multidimenional numpy array which holds the entries of the tensor

    Examples
    --------
    Create dense tensor from numpy array

    >>> T = np.zeros((3, 4, 2))
    >>> T[:, :, 0] = [[ 1,  4,  7, 10], [ 2,  5,  8, 11], [3,  6,  9, 12]]
    >>> T[:, :, 1] = [[13, 16, 19, 22], [14, 17, 20, 23], [15, 18, 21, 24]]
    >>> T = dtensor(T)
    """

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def __eq__(self, other):
        return np.equal(self, other)

    def _ttm_compute(self, V, mode, transp):
        sz = np.array(self.shape)
        r1, r2 = from_to_without(0, self.ndim, mode, separate=True)
        order = [mode] + r1 + r2
        newT = self.transpose(axes=order)
        newT = newT.reshape(sz[mode], np.prod(sz[r1 + list(range(mode + 1, len(sz)))]))
        if transp:
            newT = V.T.dot(newT)
            p = V.shape[1]
        else:
            newT = V.dot(newT)
            p = V.shape[0]
        newsz = [p] + list(sz[:mode]) + list(sz[mode + 1:])
        newT = newT.reshape(newsz)
        # transpose + argsort(order) equals ipermute
        newT = newT.transpose(np.argsort(order))
        return dtensor(newT)

    def _ttv_compute(self, v, dims, vidx, remdims):
        """
        Tensor times vector product

        Parameter
        ---------
        """
        if not isinstance(v, tuple):
            raise ValueError('v must be a tuple of vectors')
        ndim = self.ndim
        order = list(remdims) + list(dims)
        if ndim > 1:
            T = self.transpose(order)
        sz = np.array(self.shape)[order]
        for i in np.arange(len(dims), 0, -1):
            T = T.reshape((sz[:ndim - 1].prod(), sz[ndim - 1]))
            T = T.dot(v[vidx[i - 1]])
            ndim -= 1
        if ndim > 0:
            T = T.reshape(sz[:ndim])
        return T

    def ttt(self, other, modes=None):
        pass

    def unfold(self, mode):
        """
        Unfolds a dense tensor in mode n.

        Parameters
        ----------
        mode : int
            Mode in which tensor is unfolded

        Returns
        -------
        unfolded_dtensor : unfolded_dtensor object
            Tensor unfolded along mode

        Examples
        --------
        Create dense tensor from numpy array

        >>> T = np.zeros((3, 4, 2))
        >>> T[:, :, 0] = [[ 1,  4,  7, 10], [ 2,  5,  8, 11], [3,  6,  9, 12]]
        >>> T[:, :, 1] = [[13, 16, 19, 22], [14, 17, 20, 23], [15, 18, 21, 24]]
        >>> T = dtensor(T)

        Unfolding of dense tensors

        >>> T.unfold(0)
        unfolded_dtensor([[ 1.,  4.,  7., 10., 13., 16., 19., 22.],
                          [ 2.,  5.,  8., 11., 14., 17., 20., 23.],
                          [ 3.,  6.,  9., 12., 15., 18., 21., 24.]])
        >>> T.unfold(1)
        unfolded_dtensor([[ 1.,  2.,  3., 13., 14., 15.],
                          [ 4.,  5.,  6., 16., 17., 18.],
                          [ 7.,  8.,  9., 19., 20., 21.],
                          [10., 11., 12., 22., 23., 24.]])
        >>> T.unfold(2)
        unfolded_dtensor([[ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.,
                           12.],
                          [13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23.,
                           24.]])
        """

        sz = np.array(self.shape)
        N = len(sz)
        order = ([mode], from_to_without(N - 1, -1, mode, step=-1, skip=-1))
        newsz = (sz[order[0]][0], np.prod(sz[order[1]]))
        arr = self.transpose(axes=(order[0] + order[1]))
        arr = arr.reshape(newsz)
        return unfolded_dtensor(arr, mode, self.shape)

    def norm(self):
        """
        Computes the Frobenius norm for dense tensors
        :math:`\\| X \\| = \\sqrt{\\sum_{i_1,\\ldots,i_N} x_{i_1,\\ldots,i_N}^2}`

        References
        ----------
        [Kolda and Bader, 2009; p.457]
        """
        return np.linalg.norm(self)

    @inherit_docstring_from(tensor_mixin)
    def uttkrp(self, U, n):
        order = list(range(n)) + list(range(n + 1, self.ndim))
        Z = khatrirao(tuple(U[i] for i in order), reverse=True)
        return self.unfold(n).dot(Z)

    @inherit_docstring_from(tensor_mixin)
    def transpose(self, axes=None):
        return dtensor(np.transpose(np.array(self), axes=axes))


class unfolded_dtensor(np.ndarray):

    def __new__(cls, input_array, mode, ten_shape):
        obj = np.asarray(input_array).view(cls)
        obj.ten_shape = ten_shape
        obj.mode = mode
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.ten_shape = getattr(obj, 'ten_shape', None)
        self.mode = getattr(obj, 'mode', None)

    def fold(self):
        shape = np.array(self.ten_shape)
        N = len(shape)
        order = ([self.mode], from_to_without(0, N, self.mode, reverse=True))
        arr = self.reshape(tuple(shape[order[0]],) + tuple(shape[order[1]]))
        arr = np.transpose(arr, np.argsort(order[0] + order[1]))
        return dtensor(arr)
