
# -*- coding: utf-8 -*-

from __future__ import division, print_function

# This package
import data
from kernels import kernel_matrix, rbf, median_distance, poly
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


def main(dataset='magic'):
    """
    Run experiments for determining the number of principal components to
    retain in kernel PCA through cross-validation.

    After each plot is shown the program halts, close the plot to continue.

    Parameters
    ----------
    dataset : str
        Either 'magic', 'yeast', 'cardiotocography' or 'segmentation'

    """

    if not dataset in ('magic', 'yeast', 'cardiotocography', 'segmentation'):
        raise ValueError("Unknown dataset.")

    X = getattr(data, "get_" + dataset + "_data")()

    for datasize, n_iter in zip((10, 50, 100), (10, 50, 90)):

        X_i = X[:datasize]

        sigma = median_distance(X_i)
        kernel = lambda x, y: rbf(x, y, sigma)
        kpca_cv_experiment(X_i, kernel, dataset, n_iter, "rbf")

        kpca_cv_experiment(X_i, poly, dataset, n_iter, "polynomial")


def kpca_cv_experiment(X, kernel, dataset, n_iter, kernel_label):
    """
    """

    print("\nCross-validation of kernel PCA\n------------------------------")

    K = kernel_matrix(X, kernel)
    n = K.shape[0]

    print("Number of data points: {}".format(n))
    print("Kernel: {}".format(kernel_label))

    pc, errs = kpca_cv(K, n_iter)

    print("Selected PC: {}".format(pc))

    err_mean = errs.mean(0)
    title = dataset + " " + kernel_label
    plotting(np.arange(n_iter-1)+1, err_mean, title, "k", "Mean error")


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
