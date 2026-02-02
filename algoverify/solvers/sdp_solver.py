from algoverify.solvers.solver import Solver


class SDPSolver(Solver):
    """
    Abstract base class for Semidefinite Programming (SDP) solvers.

    This class extends the base Solver class to provide SDP-specific functionality
    for verification of first-order optimization methods. It serves as the parent
    class for concrete SDP solver implementations.

    The SDPSolver is currently a skeleton class that will be extended in future
    releases to support:
    - SDP formulations of verification problems
    - Semidefinite relaxations
    - Matrix variable handling
    - SDP-specific constraint generation

    Parameters
    ----------
    solver_params : dict, optional
        Solver-specific parameters and configuration options.

    Notes
    -----
    This is a placeholder class. SDP-specific functionality will be added in a
    separate pull request.
    """

    def __init__(self, solver_params=None):
        super(SDPSolver, self).__init__(solver_params=solver_params)
