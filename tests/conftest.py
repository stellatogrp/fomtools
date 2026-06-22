import numpy as np
import pytest

from fomtools.verifier import Verifier

# available solvers used as parameters to initialize verifier objects in each fixture
SOLVERS = ["gurobi", "highs"]

# SDP solvers for parameterized testing
SDP_SOLVERS = ["clarabel", "cvxpy"]

# Test tolerance for numerical comparisons
TEST_EPS = 1e-5

# Tolerance for comparing against the cvxpy reference prox. The reference proxes solve a QP
# with cvxpy's default conic solver, which is only accurate to ~5e-4 on these problems, so
# equality comparisons against them (and MIP-vs-reference checks) use this looser tolerance.
PROX_ATOL = 1e-3

# Native MIP solver options per backend (no cross-solver name translation).
_MIP_SILENT = {"gurobi": {"OutputFlag": 0}, "highs": {"output_flag": False}}
_MIP_LIMITED = {
    "gurobi": {"OutputFlag": 0, "TimeLimit": 10, "MIPGap": 1e-4},
    "highs": {"output_flag": False, "time_limit": 10, "mip_rel_gap": 1e-4},
}


def mip_params(solver, *, limited=False):
    """Return native MIP solver options for ``solver`` (silences output).

    Tests are parameterized over both Gurobi and HiGHS, which use different
    native option names. This helper returns the right dict per backend so no
    name translation is needed at the solver layer.
    """
    return dict((_MIP_LIMITED if limited else _MIP_SILENT)[solver])


@pytest.fixture(params=SOLVERS, scope="function")
def vp(request: pytest.FixtureRequest):
    """
    Pytest fixture that provides a `Verifier` instance. This fixture
    is parameterized to run tests using different solvers.
    """
    return Verifier(solver=request.param, obbt_solver=request.param)


@pytest.fixture(params=SOLVERS, scope="function")
def vp_param_1(request: pytest.FixtureRequest):
    """
    Pytest fixture that provides a `Verifier` instance configured with
    a solver whose output flag is set to 0 (silent mode).
    """
    return Verifier(
        solver=request.param, obbt_solver=request.param, params=mip_params(request.param)
    )


@pytest.fixture(params=SOLVERS, scope="function")
def vp_param_2(request: pytest.FixtureRequest):
    """
    Pytest fixture that provides a `Verifier` instance whose solver is
    configured with a 10 seconds time limit, a MIP gap set of 1e-4, and the
    output flag set to 0 (silent mode).
    """
    return Verifier(
        solver=request.param,
        obbt_solver=request.param,
        params=mip_params(request.param, limited=True),
    )


@pytest.fixture(params=SOLVERS, scope="function")
def vp_theory(request: pytest.FixtureRequest):
    """
    Pytest fixture that provides a `Verifier` object initialized with a
    custom theoretical function, `theory_func`
    """

    def theory_func(k):
        init_C = 0.5
        if k == 1:
            return np.inf
        return 2 * init_C / np.sqrt((k - 1) * (k + 2))

    return Verifier(solver=request.param, obbt_solver=request.param, theory_func=theory_func)


@pytest.fixture(params=SDP_SOLVERS, scope="function")
def vp_sdp(request: pytest.FixtureRequest):
    """
    Pytest fixture that provides an SDP Verifier instance.
    Parameterized to run tests with different SDP solver backends.
    """
    return Verifier(method="sdp", solver=request.param, obbt_solver=None)


@pytest.fixture(params=SDP_SOLVERS, scope="function")
def vp_sdp_obbt(request: pytest.FixtureRequest):
    """
    Pytest fixture providing an SDP Verifier with the OBBT handler enabled
    (using HiGHS as the LP backend for bound tightening).
    """
    return Verifier(method="sdp", solver=request.param, obbt_solver="highs")


# --------------------------------------------------------------------------
# Shared builders/helpers for OBBT tests.
# --------------------------------------------------------------------------


def build_lasso_ista(obbt, solver, *, method="mip", K=3, seed=0, n=3, m=4, t=0.05, lambd=0.1):
    """
    Build a K-step ISTA (lasso) soft-threshold chain, mirroring
    ``_obbt_leak_repro.py``.

    Parameters
    ----------
    obbt : str or None
        OBBT handler solver name (e.g. ``"gurobi"``/``"highs"``) or ``None``.
    solver : str
        Main backend solver name.
    method : str, default "mip"
        Verifier backend method (``"mip"`` or ``"sdp"``).

    Returns
    -------
    tuple[Verifier, list[LinearExpression]]
        The verifier and the list of iterates ``[z_0, ..., z_K]``.
    """
    np.random.seed(seed)
    A = np.random.normal(size=(m, n))
    W = np.eye(n) - t * A.T @ A
    G = t * A.T

    params = mip_params(solver) if method == "mip" else None
    v = Verifier(method=method, solver=solver, obbt_solver=obbt, params=params)
    z = v.add_iterate(n, lb=-0.5 * np.ones(n), ub=0.5 * np.ones(n))
    b = v.add_param(m, lb=-1.0 * np.ones(m), ub=1.0 * np.ones(m))

    iterates = [z]
    for _ in range(K):
        z = v.step.soft_threshold(arg=W @ z + G @ b, lambd=lambd * t)
        iterates.append(z)
    return v, iterates


def step_input_widths(v, step_name="SoftThreshold"):
    """Return per-step input bound widths (``ub - lb``) for steps of a type."""
    steps = [s for s in v.canonicalizer.steps if type(s).__name__ == step_name]
    return [s.ub - s.lb for s in steps]
