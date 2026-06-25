"""The cvxpygen solver interface for fomtools-generated FOM solvers.

:class:`FomtoolsInterface` plugs into cvxpygen as the solver named
``'FOMTOOLS'``: it reuses cvxpygen's QP canonicalization (registered there
under ``cvxpy_solver_name = 'OSQP'``) and, in :meth:`generate_code`, runs the
fomtools code generator to produce the embedded solver inside cvxpygen's
``c/solver_code/`` directory, plus the C bridge from
:mod:`fomtools.cvxpygen.glue`.

cvxpygen has no public solver registry — its dispatch is a dict local to
``Canonicalizer._get_interface_class`` — so :func:`register` wraps that method
(idempotently) to add the ``'FOMTOOLS'`` name without forking cvxpygen.
Generation options that cvxpygen's plumbing does not carry (FOM choice,
fomtools config, agent driver, parameter bounds) travel through a
context-variable set by :func:`fomtools.cvxpygen.api.generate_solver`.

Validated against cvxpygen 1.x (see the seam-canary test in
``tests/codegen/test_cvxpygen_register.py``).
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from cvxpygen.canonicalizer import Canonicalizer
from cvxpygen.mappings import Setting, WorkspacePointerInfo
from cvxpygen.solvers import QPCanonMixin, SolverInterface

from fomtools.codegen.agent.driver import AgentDriver
from fomtools.codegen.config import Backend, CodegenConfig, Version
from fomtools.codegen.generate import CodegenResult
from fomtools.codegen.generate import generate_code as fomtools_generate_code
from fomtools.codegen.targets import get_target
from fomtools.cvxpygen.builders import resolve_builder
from fomtools.cvxpygen.glue import cmake_solver_sources, emit_bridge, write_solver_cmake_stub
from fomtools.cvxpygen.qpdata import extract_qp_data

#: (version, backend) combinations the cvxpygen integration can emit a build for.
_SUPPORTED_TARGETS = frozenset(
    {
        (Version.EMBEDDED, Backend.C),
        (Version.LARGE_SCALE, Backend.OPENMP),
        (Version.LARGE_SCALE, Backend.CUDA),
    }
)


@dataclass
class FomtoolsOptions:
    """Options for one FOMTOOLS generation run (set by ``generate_solver``)."""

    fom: str | Callable = "admm"
    fom_options: dict[str, Any] = field(default_factory=dict)
    config: CodegenConfig = field(default_factory=CodegenConfig)
    driver: AgentDriver | None = None
    param_bounds: dict | None = None
    bound_margin: float = 1.0


_OPTIONS: contextvars.ContextVar[FomtoolsOptions | None] = contextvars.ContextVar(
    "fomtools_cvxpygen_options", default=None
)
_RESULT: contextvars.ContextVar[CodegenResult | None] = contextvars.ContextVar(
    "fomtools_cvxpygen_result", default=None
)


@contextlib.contextmanager
def pending_options(options: FomtoolsOptions):
    """Make ``options`` visible to FomtoolsInterface during cpg.generate_code."""
    opt_token = _OPTIONS.set(options)
    res_token = _RESULT.set(None)
    try:
        yield
    finally:
        _OPTIONS.reset(opt_token)
        _RESULT.reset(res_token)


def take_stashed_result() -> CodegenResult | None:
    """Return the CodegenResult produced inside the current pending_options block."""
    return _RESULT.get()


def _current_options() -> FomtoolsOptions:
    """Options for the active generation run (defaults when cvxpygen is driven
    directly via ``cpg.generate_code(..., solver='FOMTOOLS')``)."""
    return _OPTIONS.get() or FomtoolsOptions()


class FomtoolsInterface(QPCanonMixin, SolverInterface):
    """cvxpygen solver interface backed by the fomtools code generator."""

    solver_name = "FOMTOOLS"
    cvxpy_solver_name = "OSQP"
    supports_quad_obj = True

    # The bridge re-reads Canon_Params on every solve, so no per-parameter
    # solver update calls are needed.
    parameter_update_structure = {}
    solve_function_call = "{prefix}cpg_fom_solve()"

    header_files = ['"cpg_fom_bridge.h"']
    cmake_headers = ["${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"]
    cmake_sources = [
        "${CMAKE_CURRENT_SOURCE_DIR}/src/fom_solver.c",
        "${CMAKE_CURRENT_SOURCE_DIR}/src/fom_data.c",
        "${CMAKE_CURRENT_SOURCE_DIR}/src/fom_linsolve.c",
        "${CMAKE_CURRENT_SOURCE_DIR}/src/fom_ops.c",
        "${CMAKE_CURRENT_SOURCE_DIR}/src/cpg_fom_bridge.c",
    ]
    has_qdldl = False
    inmemory_preconditioning = False

    # The bridge statically allocates the workspace and solution arrays.
    ws_statically_allocated_in_solver_code = True
    sol_statically_allocated = True
    status_is_int = True
    numeric_types = {"float": "double", "int": "int"}

    # Settings live in cvxpygen's generated Canon_Settings struct; the bridge
    # copies them into the FomWorkspace before each solve.
    stgs_requires_extra_struct_type = True
    stgs_dynamically_allocated = False
    stgs_reset_function = None
    stgs_direct_write_ptr = None
    stgs = {
        "max_iter": Setting(type="cpg_int", default="1000"),
        "tol": Setting(type="cpg_float", default="0.0"),
        "check_termination": Setting(type="cpg_int", default="1"),
    }

    ws_ptrs = WorkspacePointerInfo(
        objective_value="fom_obj_val",
        iterations="fom_iter",
        status="fom_status",
        primal_residual="fom_pri_res",
        dual_residual="fom_dua_res",
        primal_solution="sol_x",
        dual_solution="sol_{dual_var_name}",
    )

    docu = "https://github.com/stellatogrp/fomtools"

    def __init__(self, data, p_prob, enable_settings):
        opts = _current_options()
        term = opts.config.termination
        # Instance-level settings so defaults reflect the actual configuration.
        # check_termination is runtime-tunable (the bridge copies it into the
        # FomWorkspace each solve), so callers can change the inf-norm-change
        # check interval via cpg_set_solver_check_termination without rebuilding.
        self.stgs = {
            "max_iter": Setting(type="cpg_int", default=str(term.max_iters)),
            "tol": Setting(type="cpg_float", default=repr(float(term.tol or 0.0))),
            "check_termination": Setting(type="cpg_int", default=str(term.check_termination)),
        }
        # Build sources follow the chosen target: OpenMP reuses the embedded C
        # files; CUDA emits .cu translation units. The bridge stays host C.
        self.cmake_sources = cmake_solver_sources(get_target(opts.config))
        super().__init__(data, p_prob, enable_settings)

    def generate_code(
        self, configuration, code_dir, solver_code_dir, cvxpygen_directory, canon, gradient, prefix
    ) -> None:
        if gradient:
            raise NotImplementedError("gradient code generation is not supported for FOMTOOLS")
        opts = _current_options()
        config = opts.config
        if (config.version, config.backend) not in _SUPPORTED_TARGETS:
            raise NotImplementedError(
                "the cvxpygen integration supports the embedded C target and the "
                "large-scale OpenMP / CUDA targets; got "
                f"version={config.version.value!r}, backend={config.backend.value!r}"
            )
        if config.dtype != "float64":
            raise NotImplementedError(
                "the cvxpygen integration requires dtype='float64' (cvxpygen canonical "
                "data is double precision)"
            )

        qp = extract_qp_data(canon, param_bounds=opts.param_bounds, bound_margin=opts.bound_margin)
        spec = resolve_builder(opts.fom)(qp, **opts.fom_options)
        if not spec.param_copy:
            raise ValueError(
                "no canonical vector piece (q, l, u) depends on the CVXPY parameters; "
                "there is nothing for the generated solver to update at runtime"
            )

        solver_code_dir = Path(solver_code_dir)
        result = fomtools_generate_code(
            spec.verifier,
            spec.next_state,
            out_dir=solver_code_dir,
            config=config,
            driver=opts.driver,
            state=list(spec.state),
            params=list(spec.params),
            name=spec.name,
        )
        if not result.success:
            raise RuntimeError(
                "fomtools code generation failed to produce a verified solver:\n"
                + result.translate.log
            )

        # The bridge's objective loop exists iff Canon_Params has a P member.
        has_quad_term = canon.parameter_canon.p.get("P") is not None
        emit_bridge(spec, qp, solver_code_dir, prefix, has_quad_term=has_quad_term)
        write_solver_cmake_stub(solver_code_dir)

        license_src = Path(__file__).resolve().parents[2] / "LICENSE"
        if license_src.is_file():
            shutil.copyfile(license_src, solver_code_dir / "LICENSE")

        _RESULT.set(result)


def register() -> None:
    """Add the ``'FOMTOOLS'`` solver to cvxpygen's dispatch (idempotent)."""
    if getattr(Canonicalizer._get_interface_class, "_fomtools_patched", False):
        return
    original = Canonicalizer._get_interface_class

    @functools.wraps(original)
    def _get_interface_class(self, solver_name: str):
        if solver_name.upper() == FomtoolsInterface.solver_name:
            return FomtoolsInterface
        return original(self, solver_name)

    _get_interface_class._fomtools_patched = True
    Canonicalizer._get_interface_class = _get_interface_class
