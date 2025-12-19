from algoverify.solvers.solver import Solver


class SDPSolver(Solver):
    def __init__(self, solver_params=None):
        super(SDPSolver, self).__init__(solver_params=solver_params)
