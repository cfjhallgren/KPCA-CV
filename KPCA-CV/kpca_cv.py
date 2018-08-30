
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np


def kpca_cv(K, n_iter):
    """
    Select the number of principal components to retain in kernel PCA through
    cross-validation. Calculate the MPRESS statistic for each number of
    principal components and select the number with the lowest value.

    Parameters
    ----------
    K : numpy.ndarray, 2d
        kernel matrix

    Returns
    -------
    Result : (pc, errs)
        The selected principal components and the residual variations of all
        data points with respect to different numbers of principal components

    """
    n = K.shape[0]
    errs = np.zeros((n, n_iter-1))

    for i in range(n): # loop over data points

        K_ii = K[i,i]
        K_i = np.delete(K, i, 1)
        kx = K_i[i:i+1].T
        K_i = np.delete(K_i, i, 0)
        L, U = np.linalg.eigh(K_i)
        L = L[::-1]
        U = U[:,::-1]

        f1 = 0
        f2 = 0

        for k in range(n_iter-1): # loop over PCs

            Uk = U[:,k:k+1]
            kU = kx.T.dot(Uk) * Uk.T / L[k]
            f1 += kU.dot(kx)
            f2 += kU.dot(K_i.dot(kU.T))
            err = K_ii - 2*f1 + f2
            errs[i,k] = err

    div = 1/((n - np.arange(n_iter-1))*n)
    MPRESS = errs.sum(0) * div
    pc = np.argmin(MPRESS)+1

    return pc, errs
