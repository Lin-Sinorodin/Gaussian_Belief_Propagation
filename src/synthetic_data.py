import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from .gabp_mrf import find_edges


class DataGenerator:
    @staticmethod
    def symmetric(a: np.array) -> np.array:
        """Return a symmetrical version of NumPy array a."""
        return a + a.T - np.diag(np.diag(a))

    def get_random_data(self, dim, sparcity_threshold=-0.25):
        """Return a sparse, symmetric, positive, normal distributed and normalized data matrix and observations vector"""
        A_rand = np.random.randn(dim, dim)
        A_sparce = A_rand * (A_rand < sparcity_threshold)

        A = self.symmetric(np.triu(A_sparce)) + np.diag(np.diag(A_rand))
        A = np.abs(A)
        A = A / np.sum(A)

        b = np.abs(np.random.randn(dim, 1))
        b = b / np.sum(b)
        b = b.reshape(-1)

        return A, b

    def get_sparse_tree_matrix(self, dim, sparcity_threshold=-0.25):
        """Use minimum spanning tree to convert a sparse matrix for a sparse tree matrix"""
        A, b = self.get_random_data(dim, sparcity_threshold)
        A_diag = np.diag(np.diag(A))
        A_up_triangle = np.triu(A) - A_diag
        A_sparse = csr_matrix(A_up_triangle)
        A_min_span_tree = minimum_spanning_tree(A_sparse).toarray()
        A = A_min_span_tree + A_diag
        A = self.symmetric(A)
        return A, b

    @staticmethod
    def _add_loop_to_A(A, edge1, edge2):
        """Takes a tree graph and add edge between given nodes to get a loop"""
        edges = find_edges(A)
        random_edges = np.transpose([edges[edge1], edges[edge2]])

        for edge in random_edges:
            i, j = edge[0], edge[1]
            node_val = np.abs(np.random.randn(1)[0])
            A[i][j] = node_val
            A[j][i] = node_val
        return A

    def add_loops_to_A(self, A, max_loops):
        """create loops between already exist nodes (do nothing if already connected)"""
        num_nodes = A.shape[0]

        for i in range(max_loops):
            edge1 = np.random.randint(0, num_nodes - 1)
            edge2 = np.random.randint(0, num_nodes - 1)
            A = self._add_loop_to_A(A, edge1, edge2)

        return A

    @staticmethod
    def cut_random_edges(A, max_cut_edges):
        """cut random edges from a given graph matrix A"""
        num_nodes = A.shape[0]
        edges = find_edges(A)

        for _ in range(max_cut_edges):
            edge_idx = np.random.randint(0, num_nodes - 1)
            edge = edges[edge_idx]
            i, j = edge[0], edge[1]
            A[i][j] = 0
            A[j][i] = 0

        return A
