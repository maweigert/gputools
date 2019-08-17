import numpy as np


def accum(subs, vals, func=np.sum, issorted=False, with_subs=False):
    """
    NumPy implementation for Matlab's accumarray
    """
    # sort accmap for ediff if not sorted
    if not issorted:
        sidx = np.lexsort(subs, axis=0)
        subs = [sub[sidx] for sub in subs]
        vals = vals[sidx]
    idx = np.where(np.diff(subs).any(axis=0))[0] + 1
    idx = np.concatenate(([0], idx, [subs[0].shape[0]]))

    # create values array
    nvals = np.zeros(len(idx) - 1)
    for i in range(len(idx) - 1):
        nvals[i] = func(vals[idx[i]:idx[i + 1]])

    # return results
    if with_subs:
        return nvals, tuple(sub[idx[:-1]] for sub in subs)
    else:
        return nvals


def unravel_dimension(shape, idx):
    if isinstance(idx, type(1)):
        idx = np.array([idx])
    k = [1] + list(np.cumprod(shape[:-1]))
    n = len(shape)
    subs = np.zeros((len(idx), n), dtype=np.int)
    for i in np.arange(n - 1, -1, -1):
        subs[:, i] = np.floor(idx / k[i])
        idx = idx % k[i]
    return subs
