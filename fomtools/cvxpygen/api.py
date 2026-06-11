"""User-facing entry point for the cvxpygen integration."""

from __future__ import annotations

from typing import Any, Callable

from fomtools.codegen.agent.driver import AgentDriver
from fomtools.codegen.config import CodegenConfig
from fomtools.codegen.generate import CodegenResult
from fomtools.cvxpygen.interface import (
    FomtoolsOptions,
    pending_options,
    register,
    take_stashed_result,
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
        Built-in FOM name (``"admm"`` — default, any QP — or ``"pdhg"`` — LPs,
        factorization-free) or a custom builder callable
        ``f(qp: QPData, **fom_options) -> FomSpec``.
    fom_options:
        Extra keyword arguments forwarded to the FOM builder (e.g. ``rho``,
        ``alpha`` for ADMM; ``step_scale`` for PDHG).
    config:
        fomtools :class:`~fomtools.codegen.config.CodegenConfig`; supplies the
        termination criterion (``max_iters``/``tol``) among others. Only the
        embedded C target with ``dtype='float64'`` is supported.
    driver:
        fomtools coding-agent driver. ``None`` uses the default (the Claude
        agent when ``FOMTOOLS_CODEGEN_AGENT=1`` is set; otherwise an error
        explains the options).
    param_bounds:
        Mapping from CVXPY ``Parameter`` objects (or names) to ``(lb, ub)``
        used for verification sampling. Parameters without bounds fall back to
        their current value ``+/- bound_margin`` with a warning.
    prefix, wrapper, enable_settings:
        Forwarded to ``cvxpygen.cpg.generate_code``. ``wrapper=True`` compiles
        the Python extension module so ``problem.solve(method='CPG')`` works.

    Returns
    -------
    CodegenResult
        The fomtools code-generation result (IR, timing, verification log).
    """
    register()
    from cvxpygen import cpg

    options = FomtoolsOptions(
        fom=fom,
        fom_options=dict(fom_options or {}),
        config=config or CodegenConfig(),
        driver=driver,
        param_bounds=param_bounds,
        bound_margin=bound_margin,
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
    if result is None:  # pragma: no cover - cpg should have raised already
        raise RuntimeError("cvxpygen did not invoke the FOMTOOLS solver interface")
    return result
