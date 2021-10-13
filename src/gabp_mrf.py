from numba import njit
import numpy as np


@njit(fastmath=True)
def l2_dist(mat1: np.array, mat2: np.array) -> np.array:
    """Return the L2 distance between two tensors"""
    return np.power(np.sum(np.power(mat1 - mat2, 2)), 1/2)


@njit(fastmath=True)
def find_neighbors_index(A: np.array) -> list:
    """return a list with the neighbors for each node"""
    return [np.where(A[:, i])[0] for i in range(A.shape[0])]


@njit(fastmath=True)
def find_edges(A):
    """Return 2d numpy array with all the edges"""
    node_i, node_j = np.where(A > 0)
    non_diag = np.where(node_i - node_j)[0]
    node_i = node_i[non_diag]
    node_j = node_j[non_diag]
    return np.stack((node_i, node_j), axis=0).T


@njit(fastmath=True)
def initialize_m_ij(A, b):
    P_ii = np.diag(A)
    mu_ii = b / P_ii
    P_ij = np.zeros_like(A)
    mu_ij = np.zeros_like(A)
    return P_ii, mu_ii, P_ij, mu_ij


@njit(fastmath=True)
def calc_m_ij(i, j, A, P_ii, mu_ii, P_ij, mu_ij, N_i):
    """Calculate an updated message for the gaBP algorithm"""
    A_ij = A[i][j]

    # P_i\j = p_ii + ∑_{k ∈ N(i)\j} Pₖᵢ
    P_i_wout_j = P_ii[i] + np.sum(P_ij[N_i[i], i]) - P_ij[j][i]

    # Pᵢⱼ = -Aᵢⱼ^2 * P_i\j^(-1)
    P_ij_ij = -1 * A_ij*A_ij / P_i_wout_j

    # μ_i\j = P_i\j ^ (-1)  * (pᵢᵢμᵢᵢ + ∑_{k ∈ N(i)\j} Pₖᵢμₖᵢ)
    P_mu_ii = P_ii[i] * mu_ii[i]
    P_mu_ij = P_ij[N_i[i], i] * mu_ij[N_i[i], i]
    mu_i_wout_j = (P_mu_ii + np.sum(P_mu_ij) - P_ij[j][i]*mu_ij[j][i]) / P_i_wout_j

    # μᵢⱼ = -Pᵢⱼ^(-1)*Aᵢⱼ*μ_i\j
    mu_ij_ij = -1 * A_ij * mu_i_wout_j / P_ij_ij

    return P_ij_ij, mu_ij_ij


@njit(fastmath=True)
def calc_node_marginal(i, P_ii, mu_ii, P_ij, mu_ij, N_i):
    """Get the marginal precision and the marginal mean for a given node"""
    # Pᵢ = pᵢᵢ + ∑_{k ∈ N(i)} Pₖᵢ  (marginal_precision)
    P_i = P_ii[i] + np.sum(P_ij[N_i[i],i])

    # μᵢ = (pᵢᵢμᵢᵢ + ∑_{k ∈ N(i)} Pₖᵢμₖᵢ) / Pᵢ  (marginal_mean)
    P_mu_ii = P_ii[i] * mu_ii[i]
    P_mu_ij = P_ij[N_i[i], i] * mu_ij[N_i[i], i]
    mu_i = (P_mu_ii + np.sum(P_mu_ij)) / P_i

    return P_i, mu_i


@njit(fastmath=True)
def run_GaBP(A, b, max_iter=100, convergence_threshold=1e-5):
    """Perform the GaBP algorithm on a given data matrix A and observation vector b"""
    # initialize
    P_ii, mu_ii, P_ij, mu_ij = initialize_m_ij(A, b)

    # get all neighbors edge
    edges = find_edges(A)
    num_nodes = A.shape[0]
    N_i = find_neighbors_index(A)

    # track the distance (change) of Pᵢⱼ and μᵢⱼ between iterations
    iter_dist = np.full((A.shape[0]), np.nan)

    # run the GaBP iterations
    for iteration in range(max_iter):
        # get previous state
        prev_P_ij, prev_mu_ij = np.copy(P_ij), np.copy(mu_ij)

        # update messages over all edges
        for edge in edges:
            i, j = edge[0], edge[1]
            P_ij[i][j], mu_ij[i][j] = calc_m_ij(i, j, A, P_ii, mu_ii, P_ij, mu_ij, N_i)

        # get updated state
        curr_P_ij, curr_mu_ij = P_ij, mu_ij

        # get the change of Pᵢⱼ and μᵢⱼ between last and current iteration
        P_ij_change = l2_dist(prev_P_ij, curr_P_ij)
        mu_ij_change = l2_dist(prev_mu_ij, curr_mu_ij)
        total_change = (P_ij_change + mu_ij_change) / num_nodes
        iter_dist[iteration] = total_change

        # check if average change is good enough to early-stop the algorithm
        if total_change < convergence_threshold:
            print('=> Converged after iteration', iteration+1)
            break

    # calculate marginals
    P_i = np.zeros_like(P_ii)
    mu_i = np.zeros_like(mu_ii)
    for i in range(num_nodes):
        P_i[i], mu_i[i] = calc_node_marginal(i, P_ii, mu_ii, P_ij, mu_ij, N_i)

    return P_i, mu_i, N_i, P_ii, mu_ii, P_ij, mu_ij, iter_dist
