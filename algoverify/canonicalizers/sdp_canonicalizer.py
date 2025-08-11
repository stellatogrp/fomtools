import numpy as np

from algoverify.canonicalizers.canonicalizer import Canonicalizer


class SDPCanonicalizer(Canonicalizer):

    def __init__(self, solver_params={}):
        super().__init__(solver_params=solver_params)

        self.algo_steps = []
        self.iterate_list = []
        self.param_list = []

        self.idx_range_map = {}
        self.lower_bounds = {}
        self.upper_bounds = {}

        self.range_marker = 0

        self.A_eq_matrices = []
        self.b_eq_vecs = []

        self.A_leq_matrices = []
        self.b_leq_vecs = []

        self.C = None

        self.psd_cone_handlers = []

    def get_SDP_dim(self):
        return self.range_marker + 1

    def add_param_var(self, param, lb=-np.inf, ub=np.inf):
        # TODO: add an initial iterate set through keyword arguments
        dim = param.n
        idx_bounds = (self.range_marker, self.range_marker + dim)
        self.idx_range_map[param] = [idx_bounds]
        self.lower_bounds[param] = lb
        self.upper_bounds[param] = ub
        self.range_marker += dim

        self.param_list.append(param)

    def add_param_stack_var(self, param_stack_var, param_list):
        raise NotImplementedError

    def add_initial_iterate_var(self, iterate, lb=-np.inf, ub=np.inf):
        # TODO: add an initial iterate set through keyword arguments
        dim = iterate.n
        idx_bounds = (self.range_marker, self.range_marker + dim)
        self.idx_range_map[param] = [idx_bounds]
        self.lower_bounds[param] = lb
        self.upper_bounds[param] = ub
        self.range_marker += dim

    def add_iterate_var(self, iterate, lb=-np.inf, ub=np.inf):
        dim = iterate.n
        idx_bounds = (self.range_marker, self.range_marker + dim)
        self.idx_range_map[param] = [idx_bounds]
        self.lower_bounds[param] = lb
        self.upper_bounds[param] = ub
        self.range_marker += dim

    def add_equality_constraint(self, lhs_expr, rhs_expr):
        raise NotImplementedError
