"""Built-in example FOMs for the code-generation CLI and demos.

Each builder returns a dict with at least ``verifier`` and ``next_state`` (the
iterate after one iteration), plus a ``name``. These mirror the ISTA and
Douglas-Rachford problems used throughout the test suite and use the
``simulator`` backend so no MIP/SDP solver is required.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as spa

from fomtools.verifier import Verifier


def ista_example(*, n: int = 2, m: int = 4, seed: int = 0, K: int = 3) -> dict:
    """Lasso ISTA: ``z <- soft_threshold(At z + Bt b, lambda t)``."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(m, n))
    b = rng.normal(size=(m,))
    lambd = 1.0
    t = 1.0 / np.max(np.real(np.linalg.eigvals(A.T @ A)))
    At = np.eye(n) - t * A.T @ A
    Bt = t * A.T

    v = Verifier(method="simulator", obbt_solver=None)
    b_param = v.add_param(m, lb=b - 1.0, ub=b + 1.0)
    z = v.add_iterate(n, lb=2, ub=2)
    iterates = [z]
    for _ in range(K):
        z = v.step.soft_threshold(At @ z + Bt @ b_param, lambd=lambd * t)
        iterates.append(z)
    return {"verifier": v, "next_state": iterates[1], "name": "ista"}


def dr_example(*, n: int = 4, m: int = 3, seed: int = 5, K: int = 3) -> dict:
    """Douglas-Rachford splitting for a small QP (affine solve + projection)."""
    rng = np.random.default_rng(seed)
    Phalf = rng.normal(size=(n, n))
    P = Phalf @ Phalf.T + 0.1 * np.eye(n)
    c = rng.normal(size=n)
    A = rng.normal(size=(m, n))
    b = A @ (rng.normal(size=n) / n)
    M = np.block([[P, A.T], [-A, np.zeros((m, m))]])
    lhs = spa.csc_matrix(M) + spa.eye(m + n)
    q_l = np.hstack([c - 0.1, b])
    q_u = np.hstack([c + 0.1, b])

    v = Verifier(method="simulator", obbt_solver=None)
    q_param = v.add_param(n + m, lb=q_l, ub=q_u)
    z0 = v.add_iterate(m + n, lb=np.zeros(n + m), ub=np.zeros(n + m))
    z = [z0]
    for _ in range(K):
        u = v.step.implicit_linear(z[-1] - q_param, lhs_mat=lhs.todense())
        utilde = v.step.relu(2 * u - z[-1], proj_ranges=(n, n + m))
        z.append(z[-1] + utilde - u)
    return {"verifier": v, "next_state": z[1], "name": "dr"}


EXAMPLES = {"ista": ista_example, "dr": dr_example}


def get_example(name: str) -> dict:
    if name not in EXAMPLES:
        raise ValueError(f"unknown example {name!r}; choices: {sorted(EXAMPLES)}")
    return EXAMPLES[name]()
