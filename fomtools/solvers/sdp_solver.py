from fomtools.solvers.solver import Solver


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

    def copy(self) -> "SDPSolver":
        """
        SDP solvers are stateless - copy not needed.

        SDP solver state is managed at the canonicalizer level. The actual
        solver (Clarabel, CVXPY) is instantiated fresh at solve time.

        Raises
        ------
        NotImplementedError
            SDPSolver does not support copy(). SDP state is managed at the
            canonicalizer level.
        """
        raise NotImplementedError(
            "SDPSolver does not support copy(). "
            "SDP state is managed at the canonicalizer level."
        )
