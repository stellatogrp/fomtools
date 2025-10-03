import os
import sys
import cvxpy as cp
import numpy as np

# Ensure package root is on path when running this file directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from algoverify.verifier import Verifier


def test_box_constrained_ls_fixed_point():

    m, n = 5, 3
    K = 10

    np.random.seed(0)
    A = np.random.normal(size=(m, n))
    b = np.random.normal(size=(m,))

    ATA = A.T @ A
    L = np.max(np.real(np.linalg.eigvals(ATA)))
    t = 0.5 / L

    # Box constraints l <= x <= u (vector bounds)
    l = -0.5 * np.ones(n)
    u = 0.5 * np.ones(n)

    solver_params = {
        'OutputFlag': 0,
    }
    VP = Verifier(solver_params=solver_params)

    b_offset = 0.5
    b_param = VP.add_param(m, lb=b - b_offset, ub=b + b_offset)

    # Fixed point: x* = Sat_{[l,u]}(x* - t * A^T(A x* - b))
    x_star = VP.fixed_point(
        n_dim=n,
        operator=lambda x: VP.saturated_linear_step(x - t * A.T @ (A @ x - b_param), l, u),
        lb=l,
        ub=u,
    )

    x0 = VP.add_initial_iterate(n, lb=-10, ub=10)
    x = [None for _ in range(K + 1)]
    x[0] = x0

    for k in range(1, K + 1):
        x[k] = VP.saturated_linear_step(x[k - 1] - t * A.T @ (A @ x[k - 1] - b_param), l, u)
        VP.set_infinity_norm_objective([x[k] - x_star])
        _ = VP.solve()

    x_star_val = VP.extract_sol(x_star)
    b_wc = VP.extract_sol(b_param)

    # CVXPY Box-constrained LS: minimize 0.5||Ax - b||^2 s.t. l <= x <= u
    x_opt = cp.Variable(n)
    obj = 0.5 * cp.sum_squares(A @ x_opt - b_wc)
    constraints = [x_opt >= l, x_opt <= u]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve()
    x_opt_val = x_opt.value

    f_star = 0.5 * np.sum((A @ x_star_val - b_wc) ** 2)
    f_opt = 0.5 * np.sum((A @ x_opt_val - b_wc) ** 2)
    assert abs(f_star - f_opt) <= 1e-6
    # Also verify the actual optimizer vectors coincide
    assert np.max(np.abs(x_star_val - x_opt_val)) <= 1e-6


# if __name__ == "__main__":
#     test_box_constrained_ls_fixed_point()
#     print("1 passed")


