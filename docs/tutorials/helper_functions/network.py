"""Shared problem and algorithm helpers for the network-flow tutorials."""

from __future__ import annotations

import networkx as nx
import numpy as np

from fomtools import IterateTrajectory, LinearExpression


def make_bipartite_incidence(
    n_supply: int,
    n_demand: int,
    edge_probability: float = 0.5,
    seed: int = 1,
    complete: bool = False,
):
    """Return supply/demand incidence blocks for a deterministic bipartite graph."""
    if complete:
        edges = [(i, j) for i in range(n_supply) for j in range(n_demand)]
        n_edges = len(edges)
        A_supply = np.zeros((n_supply, n_edges))
        A_demand = np.zeros((n_demand, n_edges))
        for edge_index, (i, j) in enumerate(edges):
            A_supply[i, edge_index] = 1.0
            A_demand[j, edge_index] = -1.0
        return A_supply, A_demand

    graph = nx.bipartite.random_graph(
        n_supply, n_demand, edge_probability, seed=seed, directed=False
    )
    incidence = np.asarray(nx.linalg.graphmatrix.incidence_matrix(
        graph, oriented=False
    ).todense())
    incidence[n_supply:, :] *= -1
    return incidence[:n_supply, :], incidence[n_supply:, :]


def make_network_data(
    n_supply: int = 6,
    n_demand: int = 3,
    edge_probability: float = 0.5,
    graph_seed: int = 1,
    data_seed: int = 2,
    complete: bool = False,
):
    """Return deterministic graph data for the minimum-cost flow tutorials."""
    A_supply, A_demand = make_bipartite_incidence(
        n_supply,
        n_demand,
        edge_probability=edge_probability,
        seed=graph_seed,
        complete=complete,
    )
    n_edges = A_supply.shape[1]
    supply = 10.0 * np.ones(n_supply)
    capacity = 5.0 * np.ones(n_edges)
    cost = np.random.RandomState(data_seed).uniform(5.0, 10.0, size=n_edges)
    return A_supply, A_demand, supply, capacity, cost


def pdhg_step(
    VP,
    A_supply,
    A_demand,
    eta,
    z_prev,
    v_prev,
    w_prev,
    x_param,
    capacity,
    cost,
    supply,
    k,
    momentum=False,
):
    """Apply one PDHG step, optionally with primal momentum.

    Each previous state may be a current iterate expression or an
    :class:`~fomtools.iteratetrajectory.IterateTrajectory`. Trajectories read
    slot ``k - 1`` and store the corresponding next state in slot ``k``.
    """
    def unpack(value, name):
        if isinstance(value, IterateTrajectory):
            return value, value[k - 1]
        if isinstance(value, LinearExpression):
            return None, value
        raise TypeError(
            f"{name} must be a LinearExpression or IterateTrajectory, got "
            f"{type(value).__name__}."
        )

    z_trajectory, z_current = unpack(z_prev, "z_prev")
    v_trajectory, v_current = unpack(v_prev, "v_prev")
    w_trajectory, w_current = unpack(w_prev, "w_prev")

    beta_k = (k - 1) / (k + 2) if momentum else 0.0
    z_next = VP.step.satlin(
        z_current - eta * (cost + A_supply.T @ v_current - A_demand.T @ w_current),
        l=np.zeros(A_supply.shape[1]),
        u=capacity,
    )
    if z_trajectory is not None:
        z_trajectory[k] = z_next
        z_next = z_trajectory[k]

    z_bar = z_next + beta_k * (z_next - z_current)
    v_next = VP.step.relu(
        v_current + eta * (-supply + A_supply @ (2 * z_bar - z_current))
    )
    if v_trajectory is not None:
        v_trajectory[k] = v_next
        v_next = v_trajectory[k]

    w_next = VP.step.implicit_linear(
        w_current + eta * (x_param - A_demand @ (2 * z_bar - z_current)),
        output_mat=np.eye(A_demand.shape[0]),
    )
    if w_trajectory is not None:
        w_trajectory[k] = w_next
        w_next = w_trajectory[k]

    residual = [
        (1 / eta) * (z_next - z_current)
        - A_supply.T @ (v_next - v_current)
        + A_demand.T @ (w_next - w_current),
        -(1 + 2 * beta_k) * A_supply @ (z_next - z_current)
        + (1 / eta) * (v_next - v_current),
        (1 + 2 * beta_k) * A_demand @ (z_next - z_current)
        + (1 / eta) * (w_next - w_current),
    ]
    return (
        z_trajectory if z_trajectory is not None else z_next,
        v_trajectory if v_trajectory is not None else v_next,
        w_trajectory if w_trajectory is not None else w_next,
        residual,
    )


def solve_pdhg_numpy(
    A_supply,
    A_demand,
    eta,
    capacity,
    cost,
    supply,
    z0,
    v0,
    w0,
    x,
    max_iters,
):
    """Run the non-momentum PDHG recurrence with NumPy."""
    z = np.array(z0, dtype=float, copy=True)
    v = np.array(v0, dtype=float, copy=True)
    w = np.array(w0, dtype=float, copy=True)
    for _ in range(max_iters):
        z_next = np.clip(
            z - eta * (cost + A_supply.T @ v - A_demand.T @ w), 0.0, capacity
        )
        v_next = np.maximum(v + eta * (-supply + A_supply @ (2 * z_next - z)), 0.0)
        w_next = w + eta * (x - A_demand @ (2 * z_next - z))
        z, v, w = z_next, v_next, w_next
    return np.concatenate([z, v, w])
