"""Shared builders for code-generation tests.

These construct small ISTA and Douglas-Rachford verifiers with the ``simulator``
backend (so no MIP/SDP solver license is needed) and provide matching pure-NumPy
reference implementations to validate the generated artifacts against.
"""

import shutil

import numpy as np
import pytest
import scipy.sparse as spa

from fomtools.codegen.config import CodegenConfig, TerminationKind, TerminationSpec
from fomtools.verifier import Verifier


@pytest.fixture
def requires_compiler():
    """Skip a test when no C compiler is available."""
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        pytest.skip("no C compiler available")
    return cc


# --------------------------------------------------------------------------
# ISTA
# --------------------------------------------------------------------------
def build_ista(*, K=5, seed=0, n=2, m=4, max_iters=200, tol=None, dtype="float64"):
    """Build an ISTA soft-threshold verifier and return everything codegen needs."""
    np.random.seed(seed)
    A = np.random.normal(size=(m, n))
    b = np.random.normal(size=(m,))
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

    kind = TerminationKind.INF_NORM_CHANGE if tol is not None else TerminationKind.MAX_ITERS
    config = CodegenConfig(
        termination=TerminationSpec(kind=kind, max_iters=max_iters, tol=tol),
        dtype=dtype,
    )
    data = dict(A=A, b=b, t=t, lambd=lambd, At=At, Bt=Bt, n=n, m=m)
    return dict(
        verifier=v,
        next_state=iterates[1],
        iterates=iterates,
        param=b_param,
        config=config,
        data=data,
        name="ista",
    )


def ista_reference(data, b_val, x0, n_iters, tol=None):
    """Pure-NumPy ISTA trajectory (matches the FOM defined by ``build_ista``)."""
    At, Bt, t, lambd = data["At"], data["Bt"], data["t"], data["lambd"]
    xk = np.asarray(x0, dtype=float).copy()
    traj = [xk.copy()]
    for _ in range(n_iters):
        r = At @ xk + Bt @ b_val
        new = np.sign(r) * np.maximum(np.abs(r) - lambd * t, 0.0)
        resid = np.max(np.abs(new - xk))
        xk = new
        traj.append(xk.copy())
        if tol is not None and resid <= tol:
            break
    return traj


# --------------------------------------------------------------------------
# Douglas-Rachford
# --------------------------------------------------------------------------
def build_dr(*, K=5, seed=5, n=4, m=3, max_iters=200, tol=None, dtype="float64"):
    """Build a Douglas-Rachford verifier (affine solve + partial-projection ReLU)."""
    np.random.seed(seed)
    Phalf = np.random.normal(size=(n, n))
    P = Phalf @ Phalf.T + 0.1 * np.eye(n)
    c = np.random.normal(size=n)
    A = np.random.normal(size=(m, n))
    xtest = 1.0 / n * np.random.normal(size=n)
    b = A @ xtest
    M = np.block([[P, A.T], [-A, np.zeros((m, m))]])
    lhs = spa.csc_matrix(M) + spa.eye(m + n)
    c_off = 0.1
    q_l = np.hstack([c - c_off, b])
    q_u = np.hstack([c + c_off, b])

    v = Verifier(method="simulator", obbt_solver=None)
    q_param = v.add_param(n + m, lb=q_l, ub=q_u)
    z0 = v.add_iterate(m + n, lb=np.zeros(n + m), ub=np.zeros(n + m))
    z = [z0]
    for _ in range(K):
        u = v.step.implicit_linear(z[-1] - q_param, lhs_mat=lhs.todense())
        utilde = v.step.relu(2 * u - z[-1], proj_ranges=(n, n + m))
        z.append(z[-1] + utilde - u)

    kind = TerminationKind.INF_NORM_CHANGE if tol is not None else TerminationKind.MAX_ITERS
    config = CodegenConfig(
        termination=TerminationSpec(kind=kind, max_iters=max_iters, tol=tol),
        dtype=dtype,
    )
    data = dict(lhs=lhs, n=n, m=m, c=c, b=b)
    return dict(
        verifier=v,
        next_state=z[1],
        iterates=z,
        param=q_param,
        config=config,
        data=data,
        name="dr",
    )


def build_boxqp(*, n=3, seed=2, max_iters=200, tol=None, dtype="float64"):
    """Projected-gradient on a box: ``x <- clip(x - t (P x + q), l, u)``.

    Exercises the saturated-linear step with constant *vector* clip bounds.
    """
    rng = np.random.default_rng(seed)
    half = rng.normal(size=(n, n))
    P = half @ half.T + 0.1 * np.eye(n)
    q0 = rng.normal(size=n)
    t = 1.0 / np.max(np.real(np.linalg.eigvals(P)))
    lo = -0.5 * np.ones(n)
    hi = 0.5 * np.ones(n)
    W = np.eye(n) - t * P

    v = Verifier(method="simulator", obbt_solver=None)
    q = v.add_param(n, lb=q0 - 0.5, ub=q0 + 0.5)
    x = v.add_iterate(n, lb=0, ub=0)
    iterates = [x]
    for _ in range(3):
        x = v.step.satlin(W @ x - t * q, l=lo, u=hi)
        iterates.append(x)

    kind = TerminationKind.INF_NORM_CHANGE if tol is not None else TerminationKind.MAX_ITERS
    config = CodegenConfig(
        termination=TerminationSpec(kind=kind, max_iters=max_iters, tol=tol), dtype=dtype
    )
    data = dict(P=P, q0=q0, t=t, W=W, lo=lo, hi=hi, n=n)
    return dict(
        verifier=v,
        next_state=iterates[1],
        iterates=iterates,
        param=q,
        config=config,
        data=data,
        name="boxqp",
    )


def boxqp_reference(data, q_val, x0, n_iters, tol=None):
    W, t, lo, hi = data["W"], data["t"], data["lo"], data["hi"]
    xk = np.asarray(x0, dtype=float).copy()
    traj = [xk.copy()]
    for _ in range(n_iters):
        new = np.minimum(np.maximum(W @ xk - t * q_val, lo), hi)
        resid = np.max(np.abs(new - xk))
        xk = new
        traj.append(xk.copy())
        if tol is not None and resid <= tol:
            break
    return traj


def dr_reference(data, q_val, x0, n_iters, tol=None):
    """Pure-NumPy Douglas-Rachford trajectory."""
    lhs, n = data["lhs"], data["n"]
    zk = np.asarray(x0, dtype=float).copy()
    traj = [zk.copy()]
    for _ in range(n_iters):
        u = spa.linalg.spsolve(lhs, zk - q_val)
        utilde = 2 * u - zk
        utilde[n:] = np.maximum(utilde[n:], 0.0)
        new = zk + utilde - u
        resid = np.max(np.abs(new - zk))
        zk = new
        traj.append(zk.copy())
        if tol is not None and resid <= tol:
            break
    return traj
