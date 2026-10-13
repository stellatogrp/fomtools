from fomtools.solvers.solver import Solver
from fomtools.solvers.mip_solver import MIPSolver
from fomtools.solvers.sdp_solver import SDPSolver
from fomtools.solvers.highs import HiGHSSolver
from fomtools.solvers.factory import make_solver, solver_class

_LAZY = {
    "GurobiSolver": "fomtools.solvers.gurobi",
    "ClarabelSolver": "fomtools.solvers.clarabel",
    "CuClarabelSolver": "fomtools.solvers.cuclarabel",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib

        mod = importlib.import_module(_LAZY[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
