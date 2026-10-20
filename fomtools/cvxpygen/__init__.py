"""CVXPYgen integration: model in CVXPY, solve with a fomtools-generated FOM.

This subpackage plugs the fomtools code generator into `CVXPYgen
<https://github.com/cvxgrp/cvxpygen>`_ as an external solver backend named
``'FOMTOOLS'`` — without forking cvxpygen. CVXPYgen generates the fast
canonicalization C code (user parameters -> canonical QP data) and fomtools
generates the embedded first-order solver; a small C bridge connects the two.

Typical use::

    import cvxpy as cp
    import fomtools.cvxpygen as fcpg

    q = cp.Parameter(n, name="q")
    problem = cp.Problem(...)
    fcpg.generate_solver(problem, "out/", fom="admm", param_bounds={q: (lo, hi)})

    from out.cpg_solver import cpg_solve
    problem.register_solve("CPG", cpg_solve)
    problem.solve(method="CPG")

The FOM behind ``fom=`` may be one of the built-ins (``"admm"``,
``"relaxed_admm"``, ``"pdhg"``) or a builder of your own, generated down any of
the three code-generation paths — one stationary iteration repeated, a method
whose varying coefficients are declared and tabulated, or a whole recorded run
emitted as a single pass. See `fomtools.cvxpygen.builders` and the
tutorial for how a builder declares which.

Requires the optional dependencies ``cvxpy``, ``cvxpygen``, and ``osqp``
(``pip install 'fomtools[cvxpygen]'``). All imports of cvxpygen are lazy so
that importing `fomtools.cvxpygen` itself never fails.
"""

from __future__ import annotations

__all__ = ["generate_solver", "register"]


def _require_cvxpygen() -> None:
    try:
        import cvxpygen  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "fomtools.cvxpygen requires the optional dependencies cvxpy and cvxpygen. "
            "Install them with: pip install 'fomtools[cvxpygen]'"
        ) from e


def register() -> None:
    """Register the ``'FOMTOOLS'`` solver with cvxpygen (idempotent)."""
    _require_cvxpygen()
    from fomtools.cvxpygen.interface import register as _register

    _register()


def generate_solver(*args, **kwargs):
    """Generate a CVXPYgen + fomtools embedded solver. See `fomtools.cvxpygen.api.generate_solver`."""
    _require_cvxpygen()
    from fomtools.cvxpygen.api import generate_solver as _generate_solver

    return _generate_solver(*args, **kwargs)
