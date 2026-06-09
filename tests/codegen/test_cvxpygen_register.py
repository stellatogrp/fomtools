"""Registration and seam-canary tests for the cvxpygen integration.

The canary asserts every cvxpygen internal the integration touches. If it
fails after a cvxpygen upgrade, re-verify the contracts in
fomtools/cvxpygen/interface.py and glue.py, then bump the version pin in
pyproject.toml ([project.optional-dependencies] cvxpygen).
"""

import inspect

import pytest

pytest.importorskip("cvxpy")
pytest.importorskip("cvxpygen")

from fomtools.cvxpygen.interface import FomtoolsInterface, register  # noqa: E402


def test_register_is_idempotent_and_dispatches():
    from cvxpygen.canonicalizer import Canonicalizer

    register()
    first = Canonicalizer._get_interface_class
    register()
    assert Canonicalizer._get_interface_class is first, "register() must be idempotent"

    canonicalizer = Canonicalizer(solver="FOMTOOLS")
    assert canonicalizer._get_interface_class("FOMTOOLS") is FomtoolsInterface
    assert canonicalizer._get_interface_class("fomtools") is FomtoolsInterface
    # Existing solvers still resolve, unknown ones still fail.
    from cvxpygen.solvers import OSQPInterface

    assert canonicalizer._get_interface_class("OSQP") is OSQPInterface
    with pytest.raises(ValueError, match="Unsupported solver"):
        canonicalizer._get_interface_class("NOSUCHSOLVER")


def test_seam_canary():
    """Assert the cvxpygen internals the integration relies on still exist."""
    import cvxpygen.utils as utils
    from cvxpygen.canonicalizer import Canonicalizer
    from cvxpygen.generator import Generator
    from cvxpygen.mappings import ParameterCanon, ParameterInfo, Setting, WorkspacePointerInfo
    from cvxpygen.solvers import QPCanonMixin, SolverInterface

    # Dispatch seam wrapped by register().
    assert callable(Canonicalizer._get_interface_class)

    # Interface construction contract: interface_class(data, p_prob, enable_settings).
    params = list(inspect.signature(QPCanonMixin.__init__).parameters)
    assert params == ["self", "data", "p_prob", "enable_settings"], params

    # generate_code hook signature used by Generator._run_solver_code_generation.
    params = list(inspect.signature(SolverInterface.generate_code).parameters)
    assert params == [
        "self",
        "configuration",
        "code_dir",
        "solver_code_dir",
        "cvxpygen_directory",
        "parameter_canon",
        "gradient",
        "prefix",
    ], params

    # Unknown solver names must flow through Generator._resolve_solver untouched.
    stand_in = type("G", (), {"_solver": "FOMTOOLS", "_solver_opts": {}})()
    assert Generator._resolve_solver(stand_in, problem=None) == ("FOMTOOLS", 0)

    # Canonical data fields consumed by qpdata.extract_qp_data.
    for fld in (
        "p",
        "p_id_to_mapping",
        "p_id_to_changes",
        "p_id_to_size",
        "nonzero_d",
        "is_maximization",
    ):
        assert fld in ParameterCanon.__dataclass_fields__, fld
    for fld in ("flat_usp", "col_to_name_usp", "name_to_size_usp", "name_to_shape", "names"):
        assert fld in ParameterInfo.__dataclass_fields__, fld

    # Writer conventions the bridge depends on.
    assert hasattr(utils, "write_canon_cmake")
    for fld in (
        "objective_value",
        "iterations",
        "status",
        "primal_residual",
        "dual_residual",
        "primal_solution",
        "dual_solution",
    ):
        assert fld in WorkspacePointerInfo.__dataclass_fields__, fld
    for fld in ("type", "default", "enabled"):
        assert fld in Setting.__dataclass_fields__, fld


def test_interface_class_attributes():
    assert FomtoolsInterface.cvxpy_solver_name == "OSQP"
    assert FomtoolsInterface.canon_p_ids == ["P", "q", "d", "A", "l", "u"]
    assert FomtoolsInterface.dual_var_names == ["y"]
    assert FomtoolsInterface.parameter_update_structure == {}
    assert FomtoolsInterface.ws_statically_allocated_in_solver_code
    assert FomtoolsInterface.sol_statically_allocated
