import numpy as np
import scipy.sparse as spa

from algoverify.solvers.clarabel import ClarabelSolver


class CuClarabelSolver(ClarabelSolver):

    def __init__(self, solver_params=None):
        super(CuClarabelSolver, self).__init__(solver_params=solver_params)

    def solve(self):
        import cupy
        from cupyx.scipy.sparse import csr_matrix as cucsr_matrix
        from juliacall import Main as jl
