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
from fomtools.codegen.config import Backend, CodegenConfig, TerminationKind, Version
from fomtools.codegen.generate import CodegenResult
from fomtools.codegen.generate import generate_code as fomtools_generate_code
from fomtools.codegen.targets import get_target
from fomtools.cvxpygen.builders import ScheduledFomSpec, fom_kind, resolve_builder
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
    allow_unrolled: bool = False


_OPTIONS: contextvars.ContextVar[FomtoolsOptions | None] = contextvars.ContextVar(
    "fomtools_cvxpygen_options", default=None
)
_RESULT: contextvars.ContextVar[CodegenResult | None] = contextvars.ContextVar(
    "fomtools_cvxpygen_result", default=None
)
# Parameters that fell back to default bounds during this run. Stashed here so
# ``generate_solver`` can warn from the user's call site instead of deep inside
# cvxpygen's generate_code dispatch.
_UNBOUNDED_PARAMS: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "fomtools_cvxpygen_unbounded_params", default=()
)


@contextlib.contextmanager
def pending_options(options: FomtoolsOptions):
    """Make ``options`` visible to FomtoolsInterface during cpg.generate_code."""
    opt_token = _OPTIONS.set(options)
    res_token = _RESULT.set(None)
    unbounded_token = _UNBOUNDED_PARAMS.set(())
    try:
        yield
    finally:
        _OPTIONS.reset(opt_token)
        _RESULT.reset(res_token)
        _UNBOUNDED_PARAMS.reset(unbounded_token)


def take_stashed_result() -> CodegenResult | None:
    """Return the CodegenResult produced inside the current pending_options block."""
    return _RESULT.get()


def take_unbounded_params() -> tuple[str, ...]:
    """CVXPY parameters that fell back to default bounds in the current run."""
    return _UNBOUNDED_PARAMS.get()


def _current_options() -> FomtoolsOptions:
    """Options for the active generation run (defaults when cvxpygen is driven
    directly via ``cpg.generate_code(..., solver='FOMTOOLS')``)."""
    return _OPTIONS.get() or FomtoolsOptions()


def _check_spec_matches_kind(spec, kind: str) -> None:
    """Reject a builder whose returned spec cannot go down the path it declared."""
    if kind == "scheduled":
        if not isinstance(spec, ScheduledFomSpec):
            raise TypeError(
                f"the builder declares fom_kind='scheduled' but returned "
                f"{type(spec).__name__}; the scheduled path needs a ScheduledFomSpec "
                "carrying the build function and its schedules."
            )
        return
    if isinstance(spec, ScheduledFomSpec):
        raise TypeError(
            f"the builder returned a ScheduledFomSpec but declares fom_kind={kind!r}; "
            "decorate it with @fom_builder('scheduled')."
        )
    if kind == "rolled" and spec.next_state is None:
        raise ValueError(
            "the builder declares fom_kind='rolled' but its spec has no next_state. "
            "Give the iterate after one iteration, or declare @fom_builder('unrolled') "
            "to generate the whole recorded run."
        )
    if kind == "unrolled":
        if not spec.iterates:
            raise ValueError(
                "the builder declares fom_kind='unrolled' but recorded no iterates; "
                "the unrolled path generates the run they hold."
            )
        if spec.next_state is not None:
            raise ValueError(
                "the builder declares fom_kind='unrolled' but also set next_state. "
                "Those select different paths: with next_state the recorded run is only "
                "*checked* against one repeated iteration, without it the run itself is "
                "generated. Leave next_state=None for the unrolled path."
            )


def _check_unrolled_allowed(opts: "FomtoolsOptions", config: CodegenConfig) -> None:
    """Gate the unrolled path: it is opt-in, and its horizon is baked in."""
    if not opts.allow_unrolled:
        raise ValueError(
            "this FOM generates an unrolled solver — the whole recorded run emitted as "
            "one straight-line pass, which for a realistic horizon is a very large "
            "program on an embedded target, with the iteration count fixed at code "
            "generation time (cpg_set_solver_max_iter no longer applies). Pass "
            "generate_solver(..., allow_unrolled=True) to accept that, or use a "
            "scheduled builder to get one block plus a coefficient table instead."
        )
    if config.termination.kind != TerminationKind.MAX_ITERS:
        raise ValueError(
            f"termination kind {config.termination.kind.value!r} cannot be combined with "
            "an unrolled FOM: the generated solver runs exactly the run that was "
            "recorded, so there is no per-iteration residual to stop on and "
            "cpg_set_solver_tol would have nothing to act on. Use "
            "TerminationKind.MAX_ITERS, or a rolled builder, where the criterion is "
            "evaluated every iteration."
        )


def _check_ir_matches_kind(ir, kind: str) -> None:
    """Fail if generation produced a differently-shaped solver than was declared.

    A declared-unrolled FOM that *rolls back up* is fine and expected: rolling is
    a size optimization that generation performs only after verifying it
    reproduces the recording. Every other mismatch means the settings surface and
    bridge — both chosen from the declared kind — would describe the wrong solver.
    """
    actual = "scheduled" if ir.schedules else ("unrolled" if ir.unrolled else "rolled")
    if actual == kind or (kind == "unrolled" and actual == "rolled"):
        return
    raise RuntimeError(
        f"the builder declares fom_kind={kind!r} but generation produced a {actual!r} "
        f"solver. The runtime settings and the C bridge are chosen from the declared "
        f"kind, so emitting this would ship plumbing that misdescribes the solver."
    )


def _run_codegen(spec, kind: str, out_dir: Path, config: CodegenConfig, driver) -> CodegenResult:
    """Call fomtools code generation down the path ``kind`` selects."""
    if kind == "scheduled":
        return fomtools_generate_code(
            spec.build,
            out_dir=out_dir,
            config=config,
            driver=driver,
            name=spec.name,
            schedules=spec.schedules,
        )
    return fomtools_generate_code(
        spec.verifier,
        spec.next_state,
        out_dir=out_dir,
        config=config,
        driver=driver,
        state=list(spec.state),
        params=list(spec.params),
        name=spec.name,
        # With next_state (the rolled path) this is the tape generation replays
        # to *check* that the recorded method is stationary — extraction reads
        # one iteration and repeats it, so without it a builder whose
        # coefficients vary with the iteration index would be frozen at its first
        # recorded value and silently generate a different algorithm. Without
        # next_state (the unrolled path) it is the run to generate.
        iterates=list(spec.iterates) or None,
        param_values=spec.param_values,
    )


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
    # Placeholder only: the real list is target-dependent (CUDA emits ``.cu``
    # translation units, not ``.c``) and is always set per instance in
    # ``__init__`` via ``cmake_solver_sources(get_target(...))``. Left as ``None``
    # so any access before ``__init__`` fails loudly rather than silently
    # building the wrong (``.c``-only) source set.
    cmake_sources = None
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
        _UNBOUNDED_PARAMS.set(qp.unbounded_params)
        kind = fom_kind(opts.fom)
        spec = resolve_builder(opts.fom)(qp, **opts.fom_options)
        _check_spec_matches_kind(spec, kind)
        if not spec.param_copy:
            raise ValueError(
                "no canonical vector piece (q, l, u) depends on the CVXPY parameters; "
                "there is nothing for the generated solver to update at runtime"
            )
        if kind == "unrolled":
            _check_unrolled_allowed(opts, config)

        solver_code_dir = Path(solver_code_dir)
        result = _run_codegen(spec, kind, solver_code_dir, config, opts.driver)
        if not result.success:
            raise RuntimeError(
                "fomtools code generation failed to produce a verified solver:\n"
                + result.translate.log
            )
        # Before anything is emitted around it: the bridge and the runtime
        # settings are chosen from the *declared* kind, so a solver that came out
        # a different shape would be wrapped in plumbing that misdescribes it.
        _check_ir_matches_kind(result.ir, kind)

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
