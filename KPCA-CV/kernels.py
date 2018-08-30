
# -*- coding: utf-8 -*-

from itertools import combinations_with_replacement
from copy import deepcopy

import numpy as np
from numpy import ones, dot

def rbf(x, y, sigma=100):
    """
    Radial basis functions kernel

    """
    return np.exp(-(np.sum((x-y)**2))/(sigma**2))

def poly(x, y, d=5, R=1):
    """
    Polynomial kernel

    """
    return (np.sum(x*y) + R)**d

def median_distance(X):
    """
    Median distance between pairs of a subset of data examples

    """
    n = X.shape[0]
    dist = np.zeros((n+1)*n/2)
    for k, t in enumerate(combinations_with_replacement(range(n),2)):
        dist[k] = np.sqrt(np.sum(np.power(X[t[0],:] - X[t[1],:],2)))
    sigma = np.median(dist)

    return sigma

def kernel_matrix(X, kernel):
    """
    Calculate the kernel matrix

    """
    n = X.shape[0]
    K = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            K[i,j] = kernel(X[i],X[j])

    return K

