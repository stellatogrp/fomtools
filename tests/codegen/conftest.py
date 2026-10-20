"""Shared builders for code-generation tests.

These construct small ISTA and Douglas-Rachford verifiers with the ``simulator``
backend (so no MIP/SDP solver license is needed) and provide matching pure-NumPy
reference implementations to validate the generated artifacts against.
"""

import shutil

import numpy as np
import pytest
import scipy.sparse as spa

from fomtools.codegen.config import (
    Backend,
    CodegenConfig,
    SolveMethod,
    TerminationKind,
    TerminationSpec,
    Version,
)
from fomtools.codegen.examples import fista_example, scheduled_example
from fomtools.linearexpression import stack
from fomtools.verifier import Verifier


def gate(case):
    """Kwargs that turn on ``generate_code``'s stationarity check for a fixture.

    Every builder here records several iterations and exposes them as
    ``case["iterates"]`` (entry 0 is the initial iterate), plus its single
    parameter leaf as ``case["param"]``. Passing these makes ``generate_code``
    replay the recorded tape and compare it against the rolled IR before
    emitting anything, which is what catches a non-stationary recording.
    """
    leaf = case["param"]
    lb = np.asarray(leaf.lb, dtype=float).reshape(-1)
    ub = np.asarray(leaf.ub, dtype=float).reshape(-1)
    return dict(iterates=case["iterates"][1:], param_values={leaf: 0.5 * (lb + ub)})


def importorskip_cvxpygen():
    """Import and return ``cvxpygen``, skipping the whole module if it can't load.

    ``pytest.importorskip`` only treats :class:`ImportError` as "missing". But
    importing ``cvxpygen`` eagerly pulls in its solver registry -> ``pdaqp`` ->
    ``juliacall``, which on first init tries to download Julia over the network;
    on an offline machine (e.g. an HPC compute node) that raises ``URLError`` —
    not ``ImportError`` — *during* import, which would otherwise turn a clean
    skip into a collection error. Skip on any import-time exception instead.
    """
    try:
        import cvxpygen
    except Exception as exc:  # noqa: BLE001 - any import-time failure -> skip
        pytest.skip(
            f"cvxpygen unavailable ({type(exc).__name__}: {exc})",
            allow_module_level=True,
        )
    return cvxpygen


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


# --------------------------------------------------------------------------
# Accelerated lasso, both ways: recorded as a run, and declared as a schedule
# --------------------------------------------------------------------------
def build_fista_run(*, seed=0, n=2, m=4, K=6, dtype="float64"):
    """FISTA recorded as the loop a user writes, for the recorded-run path.

    One iterate leaf and the momentum recurrence in ordinary Python, so nothing
    is declared. The horizon comes from the recording, which is why there is no
    ``max_iters`` argument here.
    """
    ex = fista_example(n=n, m=m, seed=seed, K=K)
    return dict(
        verifier=ex["verifier"],
        iterates=ex["iterates"],
        param=ex["param"],
        config=CodegenConfig(dtype=dtype),
        data=ex["data"],
        name="fista",
    )


def build_scheduled(*, seed=0, n=2, m=4, max_iters=200, tol=None, dtype="float64"):
    """The same accelerated lasso declared as a schedule: a build function plus it.

    Unlike the stationary builders this returns no ``verifier``: the momentum
    coefficient must stay free so extraction can probe the recording at several
    of its values. The state carries two iterates, which is what keeps the
    dependency cone of one iteration from reaching back through the whole tape.
    """
    ex = scheduled_example(n=n, m=m, seed=seed)
    kind = TerminationKind.INF_NORM_CHANGE if tol is not None else TerminationKind.MAX_ITERS
    config = CodegenConfig(
        termination=TerminationSpec(kind=kind, max_iters=max_iters, tol=tol),
        dtype=dtype,
    )
    return dict(
        build=ex["build"],
        schedules=ex["schedules"],
        config=config,
        data=ex["data"],
        name="scheduled",
    )


def fista_reference(data, b_val, n_iters, momentum=True, tol=None):
    """Pure-NumPy trajectory of the accelerated lasso, over the two-slot state.

    Matches what the *scheduled* IR carries, ``concatenate([y_prev, y])``. With
    ``momentum=False`` this is plain ISTA over the same data, which is what the
    generated solver degenerates to if the momentum is lost — so tests can assert
    the two are genuinely different.
    """
    At, Bt, t, lambd, z0 = data["At"], data["Bt"], data["t"], data["lambd"], data["z0"]
    y_prev = np.asarray(z0, dtype=float).copy()
    y = np.asarray(z0, dtype=float).copy()
    beta = 1.0
    traj = [np.concatenate([y_prev, y])]
    for _ in range(n_iters):
        beta_next = (1.0 + np.sqrt(1.0 + 4.0 * beta * beta)) / 2.0
        gamma = (beta - 1.0) / beta_next if momentum else 0.0
        extrap = y + gamma * (y - y_prev)
        r = At @ extrap + Bt @ b_val
        y_next = np.sign(r) * np.maximum(np.abs(r) - lambd * t, 0.0)
        state = np.concatenate([y, y_next])
        resid = np.max(np.abs(state - traj[-1]))
        y_prev, y, beta = y, y_next, beta_next
        traj.append(state)
        if tol is not None and resid <= tol:
            break
    return traj


def fista_loop_reference(data, b_val, n_iters, momentum=True):
    """The plain iterate sequence, for the one-slot *recorded run*.

    Derived from :func:`fista_reference` rather than repeating the recurrence: the
    two-slot entries are ``concatenate([y_prev, y])``, so the iterates the natural
    loop records are their second halves, minus the initial state.
    """
    n = data["n"]
    traj = fista_reference(data, b_val, n_iters, momentum=momentum)
    return [state[n:] for state in traj[1:]]


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
        u = v.step.implicit_linear(z[-1] - q_param, output_mat=lhs.todense())
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


# --------------------------------------------------------------------------
# Symmetric-indefinite (NOT quasidefinite) KKT solve  ->  pivoted LU
# --------------------------------------------------------------------------
def build_indefinite_kkt(*, n=3, seed=4, max_iters=60, tol=None, dtype="float64"):
    r"""A FOM with an affine solve on a symmetric-indefinite, *non*-quasidefinite
    matrix ``K = [[0, A^T], [A, 0]]`` (zero (1,1) block, ``A`` square nonsingular).

    ``K`` is symmetric but has an all-zero diagonal, so no-pivot LDL^T breaks down
    on the very first pivot: code generation must (and does) fall back to a
    pivoted LU factorization. Singular values of ``A`` are kept in ``[1.5, 3]`` so
    ``K^{-1}`` is a contraction and the iteration ``x <- solve(K, x + q)`` stays
    bounded.
    """
    rng = np.random.default_rng(seed)
    U, _, Vt = np.linalg.svd(rng.normal(size=(n, n)))
    s = rng.uniform(1.5, 3.0, size=n)  # sigma(A) > 1  ->  |eig(K)| > 1
    A = U @ np.diag(s) @ Vt
    K = np.block([[np.zeros((n, n)), A.T], [A, np.zeros((n, n))]])

    v = Verifier(method="simulator", obbt_solver=None)
    q = v.add_param(2 * n, lb=-np.ones(2 * n), ub=np.ones(2 * n))
    x = v.add_iterate(2 * n, lb=np.zeros(2 * n), ub=np.zeros(2 * n))
    iterates = [x]
    for _ in range(3):
        x = v.step.implicit_linear(x + q, output_mat=K)
        iterates.append(x)

    kind = TerminationKind.INF_NORM_CHANGE if tol is not None else TerminationKind.MAX_ITERS
    config = CodegenConfig(
        termination=TerminationSpec(kind=kind, max_iters=max_iters, tol=tol), dtype=dtype
    )
    data = dict(K=K, A=A, n=2 * n)
    return dict(
        verifier=v,
        next_state=iterates[1],
        iterates=iterates,
        param=q,
        config=config,
        data=data,
        name="indef_kkt",
    )


# --------------------------------------------------------------------------
# PDHG (Chambolle-Pock) for a linear program  (no restarts)
# --------------------------------------------------------------------------
def build_pdhg_lp(*, n=3, m=2, seed=7, max_iters=200, tol=None, dtype="float64"):
    r"""Plain PDHG for the LP ``min c'x  s.t.  A x >= b,  l <= x <= u``.

    Primal-dual (saddle) form ``min_{l<=x<=u} max_{y>=0} c'x + y'(b - A x)``;
    one PDHG iteration (primal step, then dual step with primal extrapolation):

        x+   = clip(x - tau (c - A' y), l, u)            # box projection (satlin)
        xbar = 2 x+ - x
        y+   = relu(y + sigma (b - A xbar))              # y >= 0 projection (relu)

    Pure first-order: matrix-vector products + projections, no linear solve.
    The objective ``c`` is the solve-time parameter; ``A, b, l, u`` are fixed.
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(m, n))
    x_feas = rng.uniform(-0.5, 0.5, size=n)  # feasible point inside the box
    b = A @ x_feas - 0.5  # slack >= 0, so A x_feas >= b (feasible, bounded LP)
    c_nom = rng.normal(size=n)
    lo, hi = -np.ones(n), np.ones(n)
    normA = float(np.sqrt(np.max(np.real(np.linalg.eigvals(A.T @ A)))))
    tau = sigma = 0.9 / normA  # tau*sigma*||A||^2 < 1 (PDHG step-size condition)

    v = Verifier(method="simulator", obbt_solver=None)
    c = v.add_param(n, lb=c_nom - 0.1, ub=c_nom + 0.1)
    x = v.add_iterate(n, lb=np.zeros(n), ub=np.zeros(n))
    y = v.add_iterate(m, lb=np.zeros(m), ub=np.zeros(m))
    iterates = [stack([x, y])]
    for _ in range(3):
        x_next = v.step.satlin(x - tau * c + tau * (A.T @ y), l=lo, u=hi)
        xbar = 2 * x_next - x
        y_next = v.step.relu(y + sigma * b - sigma * (A @ xbar))
        x, y = x_next, y_next
        iterates.append(stack([x, y]))

    kind = TerminationKind.INF_NORM_CHANGE if tol is not None else TerminationKind.MAX_ITERS
    config = CodegenConfig(
        termination=TerminationSpec(kind=kind, max_iters=max_iters, tol=tol), dtype=dtype
    )
    data = dict(A=A, b=b, c_nom=c_nom, lo=lo, hi=hi, tau=tau, sigma=sigma, n=n, m=m)
    return dict(
        verifier=v,
        next_state=iterates[1],
        iterates=iterates,
        param=c,
        config=config,
        data=data,
        name="pdhg_lp",
    )


def pdhg_lp_reference(data, c_val, xy0, n_iters, tol=None):
    """Pure-NumPy PDHG-LP trajectory (matches the FOM built by build_pdhg_lp)."""
    A, b, lo, hi = data["A"], data["b"], data["lo"], data["hi"]
    tau, sigma, n = data["tau"], data["sigma"], data["n"]
    xy = np.asarray(xy0, dtype=float).copy()
    x, y = xy[:n].copy(), xy[n:].copy()
    traj = [np.concatenate([x, y])]
    for _ in range(n_iters):
        x_new = np.minimum(np.maximum(x - tau * (c_val - A.T @ y), lo), hi)
        xbar = 2 * x_new - x
        y_new = np.maximum(0.0, y + sigma * (b - A @ xbar))
        new = np.concatenate([x_new, y_new])
        resid = np.max(np.abs(new - np.concatenate([x, y])))
        x, y = x_new, y_new
        traj.append(new)
        if tol is not None and resid <= tol:
            break
    return traj


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


# --------------------------------------------------------------------------
# SPD linear-solve FOM for the CG (indirect) backend
# --------------------------------------------------------------------------
def build_spd_cg(*, n=5, seed=3, max_iters=100, dtype="float64"):
    """A FOM whose body is a single SPD linear solve: ``x <- solve(M, x + q)``.

    With ``M`` symmetric positive-definite and ``eig(M) > 1`` the outer map is a
    contraction (stable over many iterations). The affine solve is the natural
    place to exercise the CG (indirect) backend, which requires an SPD system.
    Configured for the large-scale OpenMP target with ``solve_method=CG``.
    """
    rng = np.random.default_rng(seed)
    half = rng.normal(size=(n, n))
    M = half @ half.T + 2.0 * np.eye(n)  # SPD, eigenvalues >= 2

    v = Verifier(method="simulator", obbt_solver=None)
    q = v.add_param(n, lb=-np.ones(n), ub=np.ones(n))
    x = v.add_iterate(n, lb=np.zeros(n), ub=np.zeros(n))
    iterates = [x]
    for _ in range(3):
        x = v.step.implicit_linear(x + q, output_mat=M)
        iterates.append(x)

    config = CodegenConfig(
        version=Version.LARGE_SCALE,
        backend=Backend.OPENMP,
        solve_method=SolveMethod.CG,
        termination=TerminationSpec(max_iters=max_iters),
        dtype=dtype,
    )
    data = dict(M=M, n=n)
    return dict(
        verifier=v,
        next_state=iterates[1],
        iterates=iterates,
        param=q,
        config=config,
        data=data,
        name="cgqp",
    )


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


# --------------------------------------------------------------------------
# cvxpygen integration helpers (cvxpygen imported lazily: the tests using
# this are gated by pytest.importorskip("cvxpygen"))
# --------------------------------------------------------------------------
def cvxpygen_canonicalize(problem):
    """Canonicalize a CVXPY problem to cvxpygen's QP form: (canon, interface)."""
    from cvxpygen.canonicalizer import Canonicalizer

    return Canonicalizer(solver="OSQP").canonicalize(problem)


# --------------------------------------------------------------------------
# Loops written the way a user writes them to verify: a plain Python loop over
# verifier.step.*, with no restructuring for code generation. Consumed by the
# recorded-run (unrolled) codegen tests.
# --------------------------------------------------------------------------
def _lasso_data(m=6, n=10, seed=0):
    rng = np.random.default_rng(seed)
    dmat = rng.normal(size=(m, n)) / np.sqrt(m)
    eta = 1.0 / np.max(np.linalg.eigvalsh(dmat.T @ dmat))
    return {
        "m": m,
        "n": n,
        "at": np.eye(n) - eta * dmat.T @ dmat,
        "bt": eta * dmat.T,
        "lambd": 0.05 * eta,
    }


def build_ista_loop(steps=8, trajectory=False, include_initial=False):
    """A stationary loop, written as a loop rather than as one iteration."""
    d = _lasso_data()
    v = Verifier(method="simulator", obbt_solver=None)
    x = v.add_param(d["m"], lb=-np.ones(d["m"]), ub=np.ones(d["m"]))
    z0 = v.add_iterate(d["n"], lb=np.zeros(d["n"]), ub=np.zeros(d["n"]))

    if trajectory:
        traj = v.iterate_trajectory(steps + 1)
        traj[0] = z0
        for k in range(1, steps + 1):
            traj[k] = v.step.soft_threshold(d["at"] @ traj[k - 1] + d["bt"] @ x, lambd=d["lambd"])
        return v, traj, x, d

    z, iterates = z0, [z0] if include_initial else []
    for _ in range(steps):
        z = v.step.soft_threshold(d["at"] @ z + d["bt"] @ x, lambd=d["lambd"])
        iterates.append(z)
    return v, iterates, x, d


def build_fista_loop(steps=8, momentum=True):
    """FISTA written the natural way: one iterate leaf, momentum in Python.

    With ``momentum=False`` the same code records plain ISTA, which is what a
    wrongly rolled FISTA degenerates to. Tests that need both use the toggle so
    the two recordings differ in exactly one place.
    """
    d = _lasso_data()
    v = Verifier(method="simulator", obbt_solver=None)
    x = v.add_param(d["m"], lb=-np.ones(d["m"]), ub=np.ones(d["m"]))
    s = v.add_iterate(d["n"], lb=np.zeros(d["n"]), ub=np.zeros(d["n"]))

    y_prev, y_curr, beta, iterates = s, s, 1.0, []
    for _ in range(steps):
        beta_next = (1.0 + np.sqrt(1.0 + 4.0 * beta * beta)) / 2.0
        gamma = (beta - 1.0) / beta_next if momentum else 0.0
        extrap = y_curr + gamma * (y_curr - y_prev)
        y_next = v.step.soft_threshold(d["at"] @ extrap + d["bt"] @ x, lambd=d["lambd"])
        y_prev, y_curr, beta = y_curr, y_next, beta_next
        iterates.append(y_next)
    return v, iterates, x, d


def build_halpern_loop(steps=8):
    """An anchored method: every iteration refers back to ``z_0``."""
    d = _lasso_data()
    v = Verifier(method="simulator", obbt_solver=None)
    x = v.add_param(d["m"], lb=-np.ones(d["m"]), ub=np.ones(d["m"]))
    z0 = v.add_iterate(d["n"], lb=np.zeros(d["n"]), ub=np.zeros(d["n"]))

    z, iterates = z0, []
    for k in range(steps):
        alpha = 1.0 / (k + 2.0)
        tz = v.step.soft_threshold(d["at"] @ z + d["bt"] @ x, lambd=d["lambd"])
        z = alpha * z0 + (1.0 - alpha) * tz
        iterates.append(z)
    return v, iterates, x, d


def build_mixed_step_loop(steps=8):
    """The step *type* changes with the iteration index."""
    d = _lasso_data()
    v = Verifier(method="simulator", obbt_solver=None)
    x = v.add_param(d["m"], lb=-np.ones(d["m"]), ub=np.ones(d["m"]))
    z = v.add_iterate(d["n"], lb=np.zeros(d["n"]), ub=np.zeros(d["n"]))

    iterates = []
    for k in range(steps):
        arg = d["at"] @ z + d["bt"] @ x
        z = v.step.relu(arg) if k % 2 else v.step.soft_threshold(arg, lambd=0.01 * (k + 1))
        iterates.append(z)
    return v, iterates, x, d


def build_dr_loop(steps=20, collapse_expr="auto", relaxation=None):
    """Douglas-Rachford: the iterate is carried through a non-leaf accumulation.

    ``relaxation`` takes the 0-based iteration index and returns that iteration's
    relaxation weight. Pass one to make the recording genuinely non-stationary, so
    it cannot be rolled back up.
    """
    rng = np.random.default_rng(0)
    n, m = 8, 4
    amat = rng.normal(size=(m, n))
    kkt = np.block([[np.eye(n), amat.T], [amat, -np.eye(m)]])
    dim = n + m

    v = Verifier(method="simulator", obbt_solver=None)
    q = v.add_param(dim, lb=-np.ones(dim), ub=np.ones(dim))
    z = v.iterate_trajectory(steps + 1, collapse_expr=collapse_expr)
    z[0] = v.add_iterate(dim, lb=np.zeros(dim), ub=np.zeros(dim))
    for k in range(1, steps + 1):
        u = v.step.implicit_linear(z[k - 1] - q, output_mat=kkt)
        ut = v.step.relu(2 * u - z[k - 1], proj_ranges=(n, dim))
        theta = 1.0 if relaxation is None else relaxation(k - 1)
        z[k] = z[k - 1] + theta * (ut - u)
    return v, z, q


# --------------------------------------------------------------------------
# Objective termination: the same methods, stopping on a verification objective
# --------------------------------------------------------------------------
def objective_config(*, max_iters=200, tol=1e-6, check_termination=None, dtype="float64", **kw):
    """A config stopping on the verification objective (default interval: 25)."""
    return CodegenConfig(
        termination=TerminationSpec(
            kind=TerminationKind.OBJECTIVE,
            max_iters=max_iters,
            tol=tol,
            check_termination=check_termination,
        ),
        dtype=dtype,
        **kw,
    )


def build_ista_objective(*, K=3, max_iters=200, tol=1e-6, check_termination=None, **kw):
    """ISTA stopping on its fixed-point residual ``||x_K - x_{K-1}||_inf``.

    Set the way a verification script sets it: after the loop, on the last
    recorded iteration. Locating that iteration needs the tape, so generate
    with ``gate(case)``.
    """
    dtype = kw.pop("dtype", "float64")
    case = build_ista(K=K, max_iters=max_iters, dtype=dtype)
    its = case["iterates"]
    case["verifier"].set_objective_norm(its[-1] - its[-2], np.inf)
    case["config"] = objective_config(
        max_iters=max_iters, tol=tol, check_termination=check_termination, dtype=dtype, **kw
    )
    return case


def build_scheduled_objective(*, extrapolated=True, ord=np.inf, max_iters=300, tol=1e-6, **kw):
    """The scheduled accelerated lasso, stopping on a verification objective.

    With ``extrapolated`` the objective is the prox-gradient residual at the
    extrapolated point, ``y_{k+1} - (y_k + gamma_k (y_k - y_{k-1}))``, whose
    coefficients follow the momentum schedule; otherwise it is the fixed-point
    residual ``y_{k+1} - y_k`` on the second state slot.
    """
    ex = scheduled_example()
    d = ex["data"]
    n, m, z0 = d["n"], d["m"], d["z0"]
    threshold = d["lambd"] * d["t"]

    def build(gamma):
        from fomtools.codegen.extract_scheduled import Recording

        v = Verifier(method="simulator", obbt_solver=None)
        b = v.add_param(m, lb=d["b"] - 1.0, ub=d["b"] + 1.0)
        y_prev = v.add_iterate(n, lb=z0, ub=z0)
        y_curr = v.add_iterate(n, lb=z0, ub=z0)
        extrap = y_curr + gamma * (y_curr - y_prev)
        y_next = v.step.soft_threshold(d["At"] @ extrap + d["Bt"] @ b, lambd=threshold)
        v.set_objective_norm(y_next - (extrap if extrapolated else y_curr), ord)
        return Recording(v, next_state=[y_curr, y_next], state=[y_prev, y_curr])

    return dict(
        build=build,
        schedules=ex["schedules"],
        config=objective_config(max_iters=max_iters, tol=tol, **kw),
        data=d,
        name="scheduled",
    )
