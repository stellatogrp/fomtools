"""Map user-facing solver names to solver backends.

This is the only place where a solver name is resolved to a backend, so the
canonicalizers and the verifier pass names through without knowing which backend
they select.
"""

from fomtools.solvers.highs import HiGHSSolver
from fomtools.solvers.solver import Solver


def _gurobi() -> type[Solver]:
    # gurobipy is an optional dependency, so it is imported only when requested.
    try:
        from fomtools.solvers.gurobi import GurobiSolver
    except ImportError:
        raise ImportError(
            "gurobipy is required for solver='gurobi'. Install it with: pip install fomtools[gurobi]"
        ) from None
    return GurobiSolver


_MIP_SOLVERS = {"gurobi": _gurobi, "grb": _gurobi, "highs": lambda: HiGHSSolver}
# SDP backends are instantiated by the SDP canonicalizer at solve time.
_SDP_SOLVERS = {"clarabel", "cvxpy", "cvxpy_sdp", "cuclarabel"}


def is_mip_solver(name: str) -> bool:
    """Whether ``name`` is a MIP backend (and so usable for OBBT), without importing it."""
    return name.lower() in _MIP_SOLVERS


def verbosity_flag(params: dict | None) -> bool | None:
    """
    The log setting in ``params`` under any MIP backend's verbosity key, or ``None``.

    Keys match case-insensitively, as Gurobi parameter names do. Used to carry the
    main solver's setting over to an OBBT solver of a different backend.
    """
    keys = set()
    for loader in _MIP_SOLVERS.values():
        try:
            keys.add(loader().VERBOSITY_KEY.lower())
        except ImportError:
            pass
    flags = [v for k, v in (params or {}).items() if k.lower() in keys]
    return bool(flags[-1]) if flags else None


def solver_class(name: str) -> type[Solver] | None:
    """
    The MIP solver class for ``name``, or ``None`` for an SDP or unknown backend.

    Raises ``ImportError`` if the backend's optional dependency is not installed.
    """
    loader = _MIP_SOLVERS.get(name.lower())
    return loader() if loader is not None else None


def make_solver(name: str, solver_params: dict | None = None) -> Solver | None:
    """
    Return an initialized solver instance for the requested backend.

    Parameters
    ----------
    name: str
        Case-insensitive solver name: ``"gurobi"`` (or ``"grb"``), ``"highs"``, or an
        SDP backend (``"clarabel"``, ``"cvxpy"``, ``"cvxpy_sdp"``, ``"cuclarabel"``).
    solver_params: dict or None, optional
        Native solver parameters, passed verbatim to the backend.

    Returns
    -------
    Solver or None
        The solver, or ``None`` for an SDP backend (built at solve time).

    Raises
    ------
    ValueError
        If ``name`` is not a supported solver.
    ImportError
        If the backend's optional dependency is not installed.
    """
    cls = solver_class(name)
    if cls is not None:
        return cls(solver_params)
    if name.lower() in _SDP_SOLVERS:
        return None
    raise ValueError(
        f"Unknown solver: {name}. Supported solvers: "
        "'gurobi', 'highs', 'clarabel', 'cvxpy', 'cvxpy_sdp', 'cuclarabel'."
    )
