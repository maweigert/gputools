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
from sktensor.core import khatrirao
from sktensor.dtensor import dtensor

__all__ = [
    'ktensor',
    'vectorized_ktensor',
]


class ktensor:
    """
    Tensor stored in decomposed form as a Kruskal operator.

    Intended Usage
        The Kruskal operator is particularly useful to store
        the results of a CP decomposition.

    Parameters
    ----------
    U : list of ndarrays
        Factor matrices from which the tensor representation
        is created. All factor matrices ``U[i]`` must have the
        same number of columns, but can have different
        number of rows.
    lmbda : array_like of floats, optional
        Weights for each dimension of the Kruskal operator.
        ``len(lambda)`` must be equal to ``U[i].shape[1]``

    See also
    --------
    sktensor.dtensor : Dense tensors
    sktensor.sptensor : Sparse tensors
    sktensor.ttensor : Tensors stored in form of the Tucker operator

    References
    ----------
    .. [1] B.W. Bader, T.G. Kolda
           Efficient Matlab Computations With Sparse and Factored Tensors
           SIAM J. Sci. Comput, Vol 30, No. 1, pp. 205--231, 2007
    """

    def __init__(self, U, lmbda=None):
        self.U = U
        self.shape = tuple(Ui.shape[0] for Ui in U)
        self.ndim = len(self.shape)
        self.rank = U[0].shape[1]
        self.lmbda = lmbda
        if not all(np.array([Ui.shape[1] for Ui in U]) == self.rank):
            raise ValueError('Dimension mismatch of factor matrices')
        if lmbda is None:
            self.lmbda = np.ones(self.rank)

    def __eq__(self, other):
        if isinstance(other, ktensor):
            # avoid costly elementwise comparison for obvious cases
            if self.ndim != other.ndim or self.shape != other.shape:
                return False
            # do elementwise comparison
            return all(
                [(self.U[i] == other.U[i]).all() for i in range(self.ndim)] +
                [(self.lmbda == other.lmbda).all()]
            )
        else:
            # TODO implement __eq__ for tensor_mixins and ndarrays
            raise NotImplementedError()

    def uttkrp(self, U, mode):

        """
        Unfolded tensor times Khatri-Rao product for Kruskal tensors

        Parameters
        ----------
        X : tensor_mixin
            Tensor whose unfolding should be multiplied.
        U : list of array_like
            Matrices whose Khatri-Rao product should be multiplied.
        mode : int
            Mode in which X should be unfolded.

        See also
        --------
        sktensor.sptensor.uttkrp : Efficient computation of uttkrp for sparse tensors
        ttensor.uttkrp : Efficient computation of uttkrp for Tucker operators
        """
        N = self.ndim
        if mode == 1:
            R = U[1].shape[1]
        else:
            R = U[0].shape[1]
        W = np.tile(self.lmbda, 1, R)
        for i in range(mode) + range(mode + 1, N):
            W = W * np.dot(self.U[i].T, U[i])
        return np.dot(self.U[mode], W)

    def norm(self):
        """
        Efficient computation of the Frobenius norm for ktensors

        Returns
        -------
        norm : float
            Frobenius norm of the ktensor
        """
        N = len(self.shape)
        coef = np.outer(self.lmbda, self.lmbda)
        for i in range(N):
            coef = coef * np.dot(self.U[i].T, self.U[i])
        return np.sqrt(coef.sum())

    def innerprod(self, X):
        """
        Efficient computation of the inner product of a ktensor with another tensor

        Parameters
        ----------
        X : tensor_mixin
            Tensor to compute the inner product with.

        Returns
        -------
        p : float
            Inner product between ktensor and X.
        """
        N = len(self.shape)
        R = len(self.lmbda)
        res = 0
        for r in range(R):
            vecs = []
            for n in range(N):
                vecs.append(self.U[n][:, r])
            res += self.lmbda[r] * X.ttv(tuple(vecs))
        return res

    def toarray(self):
        """
        Converts a ktensor into a dense multidimensional ndarray

        Returns
        -------
        arr : np.ndarray
            Fully computed multidimensional array whose shape matches
            the original ktensor.
        """
        A = np.dot(self.lmbda, khatrirao(tuple(self.U)).T)
        return A.reshape(self.shape)

    def totensor(self):
        """
        Converts a ktensor into a dense tensor

        Returns
        -------
        arr : dtensor
            Fully computed multidimensional array whose shape matches
            the original ktensor.
        """
        return dtensor(self.toarray())

    def tovec(self):
        v = np.zeros(np.sum([s * self.rank for s in self.shape]))
        offset = 0
        for M in self.U:
            noff = offset + np.prod(M.shape)
            v[offset:noff] = M.flatten()
            offset = noff
        return vectorized_ktensor(v, self.shape, self.lmbda)


class vectorized_ktensor:

    def __init__(self, v, shape, lmbda):
        self.v = v
        self.shape = shape
        self.lmbda = lmbda

    def toktensor(self):
        order = len(self.shape)
        rank = len(self.v) // np.sum(self.shape)
        U = [None for _ in range(order)]
        offset = 0
        for i in range(order):
            noff = offset + self.shape[i] * rank
            U[i] = self.v[offset:noff].reshape((self.shape[i], rank))
            offset = noff
        return ktensor(U, self.lmbda)

# vim: set et:
