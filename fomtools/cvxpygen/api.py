"""User-facing entry point for the cvxpygen integration."""

from __future__ import annotations

import importlib.util
import os
import shutil
import warnings
from typing import Any, Callable

from fomtools.codegen.agent.driver import AgentDriver
from fomtools.codegen.config import CodegenConfig
from fomtools.codegen.generate import CodegenResult
from fomtools.cvxpygen.interface import (
    FomtoolsOptions,
    pending_options,
    register,
    take_stashed_result,
    take_unbounded_params,
)

# Frames inside the fomtools package are skipped when locating the warning's
# call site, so the user sees their own generate_solver(...) line regardless of
# how many internal wrappers (e.g. fomtools.cvxpygen.__init__) sit in between.
_FOMTOOLS_PREFIX = os.path.dirname(os.path.dirname(__file__))


def _check_wrapper_toolchain() -> None:
    """Fail before any code generation if cvxpygen cannot build its wrapper.

    cvxpygen compiles the Python extension with ``cmake`` and installs it with
    ``python -m pip``. When either is missing it only prints the failure and the
    caller later gets ``ImportError: cannot import name 'cpg_module'`` — after
    the coding agent has already run. A ``uv`` environment has no pip unless it
    is asked for, so this is the common case.
    """
    missing = []
    if importlib.util.find_spec("pip") is None:
        missing.append("pip in this Python environment (e.g. `uv pip install pip`)")
    if shutil.which("cmake") is None:
        missing.append("cmake on PATH")
    if missing:
        raise RuntimeError(
            "generate_solver(..., wrapper=True) needs "
            + " and ".join(missing)
            + ": cvxpygen builds the Python wrapper with cmake and installs it with "
            "'python -m pip'. Install them, or pass wrapper=False to generate the C "
            "code only."
        )


def generate_solver(
    problem,
    code_dir: str = "cpg_code",
    *,
    fom: str | Callable = "admm",
    fom_options: dict[str, Any] | None = None,
    config: CodegenConfig | None = None,
    driver: AgentDriver | None = None,
    param_bounds: dict | None = None,
    bound_margin: float = 1.0,
    allow_unrolled: bool = False,
    prefix: str = "",
    wrapper: bool = True,
    enable_settings: tuple[str, ...] = (),
) -> CodegenResult:
    """Generate an embedded solver for a parametric CVXPY problem.

    CVXPYgen canonicalizes ``problem`` to the QP standard form and emits the
    fast parameter-update / canonicalization C code; fomtools generates the
    embedded first-order solver for the chosen FOM. The result is a standard
    CVXPYgen ``code_dir`` (CMake project, ``cpg_solve``, Python wrapper), with
    the fomtools solver in ``code_dir/c/solver_code/``.

    Parameters
    ----------
    problem:
        A parametric ``cvxpy.Problem`` reducible to a QP.
    code_dir:
        Output directory (cvxpygen layout).
    fom:
        Built-in FOM name — ``"admm"`` (default, any QP), ``"relaxed_admm"``
        (the same with an iteration-varying over-relaxation, generated down the
        scheduled path) or ``"pdhg"`` (LPs, factorization-free) — or a custom
        builder callable ``f(qp: QPData, **fom_options) -> FomSpec``.

        A custom builder may target any of the three generation paths by
        declaring it with :func:`~fomtools.cvxpygen.builders.fom_builder`:
        ``"rolled"`` (the default — one stationary iteration, repeated),
        ``"scheduled"`` (varying coefficients declared and tabulated, returning a
        :class:`~fomtools.cvxpygen.builders.ScheduledFomSpec`), or ``"unrolled"``
        (the whole recorded run, which also needs ``allow_unrolled=True``).
    fom_options:
        Extra keyword arguments forwarded to the FOM builder (e.g. ``rho``,
        ``alpha`` for ADMM; ``alpha_start``, ``alpha_final``, ``ramp_iters``
        for the relaxed variant; ``step_scale`` for PDHG).
    config:
        fomtools :class:`~fomtools.codegen.config.CodegenConfig`; supplies the
        termination criterion (``max_iters``/``tol``/``check_termination``)
        among others. Supported targets: the embedded C target and the
        large-scale OpenMP / CUDA targets, all with ``dtype='float64'`` (the
        canonical cvxpygen data is double precision). The OpenMP build reuses
        the embedded C sources — the data-parallel loops carry ``#ifdef
        _OPENMP`` pragmas, so it compiles and verifies serially out of the box
        and threads once the CMake build adds ``-fopenmp``. The CUDA build
        emits ``.cu`` translation units and needs a CUDA-enabled toolchain / GPU.
        ``max_iter`` is emitted as a runtime cvxpygen setting (tune it via
        ``cpg_set_solver_max_iter`` or ``problem.solve(method='CPG',
        max_iter=...)`` without regenerating) for a rolled or scheduled FOM;
        ``tol`` and ``check_termination`` are emitted too when the termination
        kind is ``INF_NORM_CHANGE``, the only kind whose generated loop checks a
        residual (under the default ``MAX_ITERS`` the check is compiled out, so
        a runtime ``tol`` is rejected rather than ignored). A scheduled one caps
        ``max_iter`` at the length of its coefficient tables, past which the
        emitted accessors would clamp and freeze the coefficients. An unrolled
        FOM exposes no runtime settings: its body is the whole run executed
        once, so there is no iteration count, tolerance or check interval to
        act on.
    driver:
        fomtools coding-agent driver. ``None`` uses the default (the Claude
        agent when ``FOMTOOLS_CODEGEN_AGENT=1`` is set; otherwise an error
        explains the options).
    param_bounds:
        Mapping from CVXPY ``Parameter`` objects (or names) to ``(lb, ub)``
        used for verification sampling. Parameters without bounds fall back to
        their current value ``+/- bound_margin`` with a warning.
    allow_unrolled:
        Accept a FOM that generates the whole recorded run as one straight-line
        pass. Opt-in because the emitted program grows with the horizon — large
        for an embedded target — and because the iteration count is then fixed at
        code-generation time, so ``cpg_set_solver_max_iter`` no longer applies.
        Ignored by rolled and scheduled builders.
    prefix, wrapper, enable_settings:
        Forwarded to ``cvxpygen.cpg.generate_code``. ``wrapper=True`` compiles
        the Python extension module so ``problem.solve(method='CPG')`` works.

    Returns
    -------
    CodegenResult
        The fomtools code-generation result (IR, timing, verification log).
    """
    if wrapper:
        _check_wrapper_toolchain()
    register()
    from cvxpygen import cpg

    options = FomtoolsOptions(
        fom=fom,
        fom_options=dict(fom_options or {}),
        config=config or CodegenConfig(),
        driver=driver,
        param_bounds=param_bounds,
        bound_margin=bound_margin,
        allow_unrolled=allow_unrolled,
    )
    with pending_options(options):
        cpg.generate_code(
            problem,
            code_dir=code_dir,
            solver="FOMTOOLS",
            prefix=prefix,
            wrapper=wrapper,
            enable_settings=list(enable_settings),
        )
        result = take_stashed_result()
        unbounded = take_unbounded_params()
    if result is None:  # pragma: no cover - cpg should have raised already
        raise RuntimeError("cvxpygen did not invoke the FOMTOOLS solver interface")
    if unbounded:
        names = ", ".join(repr(name) for name in unbounded)
        warnings.warn(
            f"no param_bounds supplied for CVXPY parameter(s) {names}; using each "
            f"parameter's current value +/- {bound_margin} for verification sampling. "
            "Pass param_bounds={parameter: (lb, ub)} to control the verified range.",
            skip_file_prefixes=(_FOMTOOLS_PREFIX,),
        )
    return result
