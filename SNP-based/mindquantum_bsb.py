#!/usr/bin/env python3
"""
MindQuantum BSB (Ballistic Simulated Bifurcation) wrapper for Max-Cut solving.
"""

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from mindquantum.algorithm.qaia import BSB


def compute_xi_from_sparse(J):
    """Compute xi parameter from coupling matrix (same logic as tabu_SB_gpu code)."""
    if not isinstance(J, csr_matrix):
        J = csr_matrix(J)
    row_sums = np.array(J.sum(axis=1)).flatten()
    max_abs_row_sum = np.max(np.abs(row_sums))
    if max_abs_row_sum == 0:
        return 1.0
    return 1.0 / max_abs_row_sum


def solve_maxcut_bsb(J, n_iter=1000, batch_size=1, dt=1.0, xi=None, h=None):
    """
    Solve max-cut problem using MindQuantum's BSB algorithm.

    Args:
        J: scipy csr_matrix or numpy array — coupling matrix (negated weight matrix).
        n_iter: number of BSB iterations.
        batch_size: number of parallel samples.
        dt: step size.
        xi: positive frequency constant. If None, computed from J.
        h: external field. Can be None, a scalar, or a 1-D array of shape (N,).

    Returns:
        best_classification: numpy array of shape (N,) with values in {-1, 1}.
        max_cut_value: float — the max-cut value of the best sample.
    """
    if not isinstance(J, csr_matrix):
        J = csr_matrix(J)

    N = J.shape[0]

    if xi is None:
        xi = compute_xi_from_sparse(J)

    # Handle h parameter
    h_array = None
    if h is not None:
        if np.isscalar(h):
            h_array = np.full((N, 1), float(h))
        else:
            h_array = np.asarray(h, dtype=float).reshape(-1, 1)

    # Create and run BSB solver
    bsb = BSB(
        J,
        h=h_array,
        x=None,
        n_iter=n_iter,
        batch_size=batch_size,
        dt=dt,
        xi=xi,
    )
    bsb.update()

    # Extract spin configuration
    spins = np.sign(bsb.x)  # shape: (N, batch_size)

    # Compute energy and cut for each sample
    # energy_k = -0.5 * s_k^T J s_k
    Js = J.dot(spins)  # (N, batch_size)
    energy = -0.5 * np.sum(Js * spins, axis=0)  # (batch_size,)

    j_sum = float(J.sum())
    cut = -0.5 * energy - 0.25 * j_sum  # (batch_size,)

    # Select best sample
    ind = int(np.argmax(cut))
    best_classification = spins[:, ind].astype(float)
    max_cut_value = float(cut[ind])

    return best_classification, max_cut_value


def build_J_from_edges(edges, nodes, negate=True):
    """
    Build coupling matrix J (scipy csr_matrix) from edge list.

    Args:
        edges: list of (node1, node2, weight) tuples.
        nodes: list of node identifiers.
        negate: whether to negate the weights (True for max-cut formulation).

    Returns:
        J: scipy csr_matrix of shape (n_v, n_v).
        node_to_idx: dict mapping node -> matrix index.
    """
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    n_v = len(nodes)

    if not edges:
        return csr_matrix((n_v, n_v)), node_to_idx

    row_indices = []
    col_indices = []
    data = []

    for edge in edges:
        if isinstance(edge, (list, tuple)) and len(edge) >= 3:
            n1, n2, w = edge[0], edge[1], float(edge[2])
        else:
            continue
        if n1 in node_to_idx and n2 in node_to_idx:
            i, j = node_to_idx[n1], node_to_idx[n2]
            val = -w if negate else w
            row_indices.extend([i, j])
            col_indices.extend([j, i])
            data.extend([val, val])

    if not data:
        return csr_matrix((n_v, n_v)), node_to_idx

    J = coo_matrix(
        (data, (row_indices, col_indices)),
        shape=(n_v, n_v)
    ).tocsr()

    return J, node_to_idx
