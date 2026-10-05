from fomtools.solvers.solver import Solver
from fomtools.solvers.mip_solver import MIPSolver
from fomtools.solvers.sdp_solver import SDPSolver
from fomtools.solvers.highs import HiGHSSolver
from fomtools.solvers.clarabel import ClarabelSolver
from fomtools.solvers.cuclarabel import CuClarabelSolver


def __getattr__(name):
    if name == "GurobiSolver":
        from fomtools.solvers.gurobi import GurobiSolver

        return GurobiSolver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
