
# -*- coding: utf-8 -*-

from __future__ import division, print_function

# This package
import data
from kernels import kernel_matrix, rbf, median_distance
from kpca_cv import kpca_cv

# Built-in modules
import sys
from time import time

# External modules
import numpy as np
from numpy import dot, diag
from matplotlib import pyplot as plt
from matplotlib import rcParams

# Matplotlib config
rcParams['font.family'] = 'serif'
rcParams['axes.titlesize'] = 17
rcParams['axes.labelsize'] = 15
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12


def main(dataset='yeast'):
    """
    Run experiments for determining the number of principal components to
    retain in kernel PCA through cross-validation.

    After each plot is shown the program halts, close the plot to continue.

    Parameters
    ----------
    dataset : str
        Either 'magic' or 'yeast'

    """

    if not dataset in ('magic', 'yeast'):
        raise ValueError("Unknown dataset.")

    X = getattr(data, "get_" + dataset + "_data")()

    kernel = lambda x, y: rbf(x, y, sigma)

    for datasize in (10, 50, 100):
        X_i = X[:datasize]
        sigma = median_distance(X_i)
        kpca_cv_experiment(X_i, kernel, dataset)


def kpca_cv_experiment(X, kernel, dataset):
    """

    """
    # TODO move actual PCA calcs to separate file

    print("\nCross-validation of kernel PCA\n------------------------------")

    K = kernel_matrix(X, kernel)
    n = K.shape[0]

    print("Number of data points: {}".format(n))

    pc, errs = kpca_cv(K)

    print("Selected PC: {}".format(pc))

    err_mean = errs.mean(0)
    plotting(np.arange(n)+1, err_mean, dataset, "k", "Mean error")


def plotting(x, y, title, xlabel, ylabel):
    plt.figure()
    plt.plot(x, y, 'b')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main(*sys.argv[1:])
