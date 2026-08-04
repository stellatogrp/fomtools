"""Tests for the backend-aware :class:`IterateTrajectory` abstraction.

Covers list semantics, the collapse-vs-leaf policy per backend, the SDP
relaxation-ordering relationship between the collapsed and no-collapse forms,
cross-backend sanity (MIP lower-bounds the SDP relaxation), the defensive
unbounded-SDP guard, and drop-in parity with the legacy ``z = [None] * (K+1)``
idiom.
"""

import numpy as np
import pytest

from fomtools import IterateTrajectory, Verifier
from fomtools.linearexpression import LinearExpression
from tests.conftest import TEST_EPS, mip_params


def _mip(solver="highs"):
    return Verifier(method="mip", solver=solver, obbt_solver=None, params=mip_params(solver))


def _sdp(solver="clarabel"):
    return Verifier(method="sdp", solver=solver, obbt_solver=None)


# --------------------------------------------------------------------------
# List semantics (backend-agnostic)
# --------------------------------------------------------------------------


def test_list_semantics():
    v = _mip()
    traj = v.iterate_trajectory(3)
    assert isinstance(traj, IterateTrajectory)
    assert len(traj) == 3

    z0 = v.add_iterate(2, lb=-1, ub=1)
    traj[0] = z0
    assert traj[0] is z0
    # negative index maps to the same slot
    assert traj[-3] is z0

    # unset slot -> LookupError
    with pytest.raises(LookupError):
        _ = traj[1]

    # out-of-range -> IndexError (both read and write)
    with pytest.raises(IndexError):
        _ = traj[3]
    with pytest.raises(IndexError):
        traj[5] = z0

    # non-LinearExpression -> TypeError
    with pytest.raises(TypeError):
        traj[1] = np.zeros(2)

    # slice access / assignment -> TypeError
    with pytest.raises(TypeError):
        _ = traj[0:2]
    with pytest.raises(TypeError):
        traj[0:2] = z0

    # iteration yields stored expressions (raises on gaps)
    traj[1] = v.step.relu(z0)
    traj[2] = v.step.relu(traj[1])
    assert [e.ndim for e in traj] == [2, 2, 2]


def test_constructor_validation():
    v = _mip()
    with pytest.raises(ValueError):
        v.iterate_trajectory(0)
    with pytest.raises(ValueError):
        v.iterate_trajectory(3, collapse_expr="sometimes")


def test_iter_raises_on_unset_slot():
    v = _mip()
    traj = v.iterate_trajectory(2)
    traj[0] = v.add_iterate(2, lb=0, ub=1)
    with pytest.raises(LookupError):
        list(traj)


# --------------------------------------------------------------------------
# Policy deltas: does an assignment mint a new leaf/step?
# --------------------------------------------------------------------------


def _affine_delta(v, collapse_expr):
    """Assign a collapsed affine expr and return (Δiterates, Δsteps)."""
    traj = v.iterate_trajectory(2, collapse_expr=collapse_expr)
    traj[0] = v.add_iterate(2, lb=-1, ub=1)
    M = np.array([[1.0, 0.5], [0.0, 1.0]])
    n_it, n_st = len(v.iterates), len(v.steps)
    traj[1] = M @ traj[0] + np.array([0.1, 0.2])
    delta = (len(v.iterates) - n_it, len(v.steps) - n_st)
    return delta, traj


def test_policy_deltas_mip():
    # auto + True collapse (no new leaf/step); False forces a linking leaf.
    assert _affine_delta(_mip(), "auto")[0] == (0, 0)
    assert _affine_delta(_mip(), True)[0] == (0, 0)
    delta, traj = _affine_delta(_mip(), False)
    assert delta == (1, 1)
    assert traj[1].is_leaf


def test_policy_deltas_sdp():
    # auto keeps a separate leaf on SDP; True forces collapse.
    delta, traj = _affine_delta(_sdp(), "auto")
    assert delta == (1, 1)
    assert traj[1].is_leaf
    assert _affine_delta(_sdp(), False)[0] == (1, 1)
    assert _affine_delta(_sdp(), True)[0] == (0, 0)


def test_leaf_value_never_mints_extra():
    # A bare leaf (relu output) is stored as-is on every backend/policy: the
    # only leaf/step added is the relu's own, none from the trajectory.
    for v, policy in [
        (_mip(), "auto"),
        (_mip(), False),
        (_sdp(), "auto"),
        (_sdp(), False),
    ]:
        traj = v.iterate_trajectory(2, collapse_expr=policy)
        traj[0] = v.add_iterate(2, lb=-1, ub=1)
        n_it, n_st = len(v.iterates), len(v.steps)
        out = v.step.relu(traj[0])
        after_step = (len(v.iterates) - n_it, len(v.steps) - n_st)
        traj[1] = out  # assigning an already-leaf value must add nothing more
        after_assign = (len(v.iterates) - n_it, len(v.steps) - n_st)
        assert after_assign == after_step
        assert traj[1] is out


# --------------------------------------------------------------------------
# SDP relaxation ordering (load-bearing correctness)
# --------------------------------------------------------------------------


def _build_softthresh_ista(solver, collapse_expr, K=2, seed=0):
    m, n = 4, 3
    t, lambd = 0.05, 0.1
    np.random.seed(seed)
    A = np.random.normal(size=(m, n))
    v = Verifier(method="sdp", solver=solver, obbt_solver=None)
    x = v.add_param(m, lb=-1, ub=1)
    z = v.iterate_trajectory(K + 1, collapse_expr=collapse_expr)
    z[0] = v.add_iterate(n, lb=-0.5, ub=0.5)
    for k in range(1, K + 1):
        z[k] = v.step.soft_threshold(z[k - 1] - t * A.T @ (A @ z[k - 1] - x), lambd=lambd * t)
    v.set_objective_norm(z[K] - z[K - 1], 2)
    return v


def test_sdp_relaxation_ordering(vp_sdp):
    """No-collapse SDP is a valid but looser relaxation: bound >= collapsed."""
    solver = vp_sdp.canonicalizer.solver_name
    b_collapse = _build_softthresh_ista(solver, True).solve()
    b_nocollapse = _build_softthresh_ista(solver, False).solve()
    b_auto = _build_softthresh_ista(solver, "auto").solve()

    # no-collapse never tightens the relaxation
    assert b_nocollapse >= b_collapse - TEST_EPS
    # 'auto' on SDP resolves to no-collapse exactly
    assert abs(b_auto - b_nocollapse) <= TEST_EPS


def _build_relu_nnls(solver, collapse_expr, K=2, seed=0):
    m, n = 3, 2
    t = 0.01
    np.random.seed(seed)
    A = np.random.normal(size=(m, n))
    v = Verifier(method="sdp", solver=solver, obbt_solver=None)
    x = v.add_param(m, lb=0, ub=0.1)
    z = v.iterate_trajectory(K + 1, collapse_expr=collapse_expr)
    z[0] = v.add_iterate(n, lb=5, ub=5.1)
    for k in range(1, K + 1):
        z[k] = v.step.relu(z[k - 1] - t * A.T @ (A @ z[k - 1] - x))
    v.set_objective_norm(z[K] - z[K - 1], 2)
    return v


def test_sdp_relu_policies_coincide(vp_sdp):
    """relu returns a leaf on SDP, so every policy gives the same bound."""
    solver = vp_sdp.canonicalizer.solver_name
    b_collapse = _build_relu_nnls(solver, True).solve()
    b_nocollapse = _build_relu_nnls(solver, False).solve()
    b_auto = _build_relu_nnls(solver, "auto").solve()
    assert abs(b_collapse - b_nocollapse) <= TEST_EPS
    assert abs(b_collapse - b_auto) <= TEST_EPS


# --------------------------------------------------------------------------
# Cross-backend sanity: MIP inf-norm lower-bounds the no-collapse SDP two-norm
# --------------------------------------------------------------------------


def test_cross_backend_mip_le_sdp(vp_sdp):
    m, n = 3, 2
    t = 0.01
    np.random.seed(0)
    A = np.random.normal(size=(m, n))
    solver = vp_sdp.canonicalizer.solver_name

    for K in [1, 2]:
        sdp_two_norm_sq = _build_relu_nnls(solver, False, K=K).solve()

        vm = _mip()
        xm = vm.add_param(m, lb=0, ub=0.1)
        zm = vm.iterate_trajectory(K + 1)  # auto -> collapse on MIP
        zm[0] = vm.add_iterate(n, lb=5, ub=5.1)
        for k in range(1, K + 1):
            zm[k] = vm.step.relu(zm[k - 1] - t * A.T @ (A @ zm[k - 1] - xm))
        vm.set_objective_norm([zm[K] - zm[K - 1]], np.inf)
        mip_inf_norm = vm.solve()

        assert mip_inf_norm <= np.sqrt(sdp_two_norm_sq) + TEST_EPS


# --------------------------------------------------------------------------
# Defensive unbounded-SDP guard (SDP rejects unbounded vars at registration,
# so build an unregistered non-leaf expression with infinite propagated bounds)
# --------------------------------------------------------------------------


def test_sdp_unbounded_guard():
    v = _sdp()
    traj = v.iterate_trajectory(2, collapse_expr=False)
    leaf = LinearExpression.from_int(2)  # default bounds are +-inf
    expr = np.eye(2) @ leaf  # non-leaf with infinite propagated bounds
    with pytest.raises(ValueError, match=r"k=1"):
        traj.assign(1, expr, collapse_expr=False)


# --------------------------------------------------------------------------
# Drop-in parity with the legacy z = [None] * (K+1) idiom on MIP
# --------------------------------------------------------------------------


def test_legacy_idiom_parity_mip(vp_param_1):
    """iterate_trajectory on MIP (auto collapses) matches the list idiom exactly."""
    m, n, K = 5, 3, 6
    np.random.seed(0)
    A = np.random.normal(size=(m, n))
    eta = 0.5 / np.max(np.linalg.eigvalsh(A.T @ A))

    def run(builder):
        v = vp_param_1
        x = v.add_param(m, lb=10, ub=11)
        z = builder(v, K + 1)
        z[0] = v.add_iterate(n, lb=1, ub=1)
        res = []
        for k in range(1, K + 1):
            z[k] = v.step.relu(z[k - 1] - eta * A.T @ (A @ z[k - 1] - x))
            v.set_objective_norm([z[k] - z[k - 1]], np.inf)
            res.append(v.solve())
        return res

    # legacy list idiom on a fresh verifier
    solver = type(vp_param_1.canonicalizer.solver).__name__
    solver = {"GurobiSolver": "gurobi", "HiGHSSolver": "highs"}[solver]

    v_list = Verifier(method="mip", solver=solver, obbt_solver=solver, params=mip_params(solver))
    xl = v_list.add_param(m, lb=10, ub=11)
    zl = [None] * (K + 1)
    zl[0] = v_list.add_iterate(n, lb=1, ub=1)
    res_list = []
    for k in range(1, K + 1):
        zl[k] = v_list.step.relu(zl[k - 1] - eta * A.T @ (A @ zl[k - 1] - xl))
        v_list.set_objective_norm([zl[k] - zl[k - 1]], np.inf)
        res_list.append(v_list.solve())

    res_traj = run(lambda v, length: v.iterate_trajectory(length))
    assert np.allclose(res_traj, res_list, atol=TEST_EPS)
