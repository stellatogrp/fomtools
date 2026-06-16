"""End-to-end test: CVXPY problem -> cvxpygen + fomtools solver -> solve('CPG').

Uses the deterministic StubAgentDriver with a checked-in golden solver body, so
no LLM is involved. To regenerate the golden after changing the test problem or
the ADMM builder, scaffold the same problem (same seeds/dimensions) and adapt
``tests/codegen/golden/cvxpygen_qp/embedded_c/fom_solver.c`` until
``make verify`` passes — or run once with FOMTOOLS_CODEGEN_AGENT=1.
"""

import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pytest

cp = pytest.importorskip("cvxpy")
from tests.codegen.conftest import importorskip_cvxpygen  # noqa: E402

importorskip_cvxpygen()

from fomtools.codegen.agent.stub import StubAgentDriver  # noqa: E402
from fomtools.codegen.config import CodegenConfig, TerminationSpec  # noqa: E402
from fomtools.cvxpygen import generate_solver  # noqa: E402

GOLDEN = Path(__file__).parent / "golden" / "cvxpygen_qp" / "embedded_c" / "fom_solver.c"


def _build_problem():
    """Fixed-seed box QP with parametric linear cost (matches the golden)."""
    rng = np.random.default_rng(42)
    n, m_eq = 3, 2
    A = rng.standard_normal((m_eq, n))
    q = cp.Parameter(n, name="q")
    q.value = 0.1 * np.ones(n)
    x = cp.Variable(n, name="x")
    problem = cp.Problem(
        cp.Minimize(0.5 * cp.sum_squares(x) + q @ x),
        [A @ x == A @ (0.1 * np.ones(n)), x <= 1.0, -1.0 <= x],
    )
    return problem, q, x


@pytest.mark.requires_compiler
@pytest.mark.parametrize("check_termination", [1, 8])
def test_generated_solver_verifies_with_tol_termination(
    requires_compiler, tmp_path, monkeypatch, check_termination
):
    """The golden body's interval-N termination must match the oracle exactly.

    Runs codegen + the C verification harness (StubAgentDriver golden) with
    tolerance-based termination at different check intervals; wrapper=False
    skips the slow CMake/pybind11 build.
    """
    from fomtools.codegen.config import TerminationKind

    monkeypatch.chdir(tmp_path)
    problem, q, x = _build_problem()
    result = generate_solver(
        problem,
        "cpg_tol",
        fom="admm",
        param_bounds={q: (-1.0, 1.0)},
        config=CodegenConfig(
            termination=TerminationSpec(
                kind=TerminationKind.INF_NORM_CHANGE,
                max_iters=4000,
                tol=1e-9,
                check_termination=check_termination,
            )
        ),
        driver=StubAgentDriver(GOLDEN),
        wrapper=False,
    )
    assert result.success


@pytest.mark.requires_compiler
def test_generate_solver_end_to_end(requires_compiler, tmp_path, monkeypatch):
    if shutil.which("cmake") is None:
        pytest.skip("cmake not available")
    if importlib.util.find_spec("pip") is None:
        pytest.skip("pip not available (cvxpygen's wrapper build runs 'python -m pip')")

    # cvxpygen's wrapper registration imports '<code_dir>.cpg_solver' relative
    # to the working directory, so use a relative code_dir.
    monkeypatch.chdir(tmp_path)
    problem, q, x = _build_problem()
    code_dir = "cpg_fomtools"

    result = generate_solver(
        problem,
        code_dir,
        fom="admm",
        param_bounds={q: (-1.0, 1.0)},
        config=CodegenConfig(termination=TerminationSpec(max_iters=2000)),
        driver=StubAgentDriver(GOLDEN),
        wrapper=True,
    )
    assert result.success

    # Generated layout: fomtools solver + bridge inside cvxpygen's solver_code.
    solver_code = tmp_path / code_dir / "c" / "solver_code"
    for rel in (
        "src/fom_solver.c",
        "src/fom_data.c",
        "src/cpg_fom_bridge.c",
        "include/cpg_fom_bridge.h",
        "CMakeLists.txt",
    ):
        assert (solver_code / rel).is_file(), rel

    # wrapper=True registered the 'CPG' solve method on the problem.
    for q_val in (0.1 * np.ones(3), np.array([0.8, -0.5, 0.3])):
        q.value = q_val
        obj_cpg = problem.solve(method="CPG")
        x_cpg = x.value.copy()
        duals_cpg = [c.dual_value.copy() for c in problem.constraints]

        obj_ref = problem.solve(solver=cp.OSQP, eps_abs=1e-9, eps_rel=1e-9)
        np.testing.assert_allclose(x_cpg, x.value, atol=1e-5)
        np.testing.assert_allclose(obj_cpg, obj_ref, atol=1e-5)
        for d_cpg, d_ref in zip(duals_cpg, [c.dual_value for c in problem.constraints]):
            np.testing.assert_allclose(d_cpg, d_ref, atol=1e-4)
