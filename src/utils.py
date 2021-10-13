import os
import sys
import numpy as np
from time import time

from .gabp_mrf import run_GaBP
from .synthetic_data import DataGenerator


class HiddenPrints:
    """
    Function to prevent code printing.
    Source: https://www.codegrepper.com/code-examples/python/python+turn+off+printing

    Example:
        with HiddenPrints():
          print("This wont print")
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def get_running_statistics(num_nodes, num_iters):
    """Run the GaBP algorithm for some iterations to get some statistics on it's performance"""
    running_time = []
    num_iter = []

    for i in range(num_iters+1):
        A, b = DataGenerator().get_sparse_tree_matrix(dim=num_nodes)

        start_time = time()
        with HiddenPrints():
            P_i, mu_i, N_i, P_ii, mu_ii, P_ij, mu_ij, iter_dist = run_GaBP(A, b)
        end_time = time()

        num_iter.append(np.sum(1-np.isnan(iter_dist)))
        dt = end_time - start_time
        running_time.append(dt)

    return np.array(running_time[1:]), np.array(num_iter[1:])
