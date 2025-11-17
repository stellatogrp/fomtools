import numpy as np

from tests.cvxpy import fixed_point as fixed_point_cvxpy


def test_box_constrained_ls_fixed_point(vp_param_1):
    m, n = 4, 2
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

    b_offset = 0.5
    b_param = vp_param_1.add_param(m, lb=b - b_offset, ub=b + b_offset)

    # Fixed point: x* = Sat_{[l,u]}(x* - t * A^T(A x* - b))
    x_star = vp_param_1.fixed_point(
        n_dim=n,
        operator=lambda x: vp_param_1.step(x - t * A.T @ (A @ x - b_param), l=l, u=u, step='satlin'),
        lb=l,
        ub=u,
    )

    x0 = vp_param_1.initial_iterate(n, lb=-10, ub=10)
    x = [None for _ in range(K + 1)]
    x[0] = x0

    for k in range(1, K + 1):
        x[k] = vp_param_1.step(x[k - 1] - t * A.T @ (A @ x[k - 1] - b_param), l=l, u=u, step='satlin')
        vp_param_1.set_infinity_norm_objective([x[k] - x_star])
        _ = vp_param_1.solve()

    x_star_val = vp_param_1.extract_sol(x_star)
    b_wc = vp_param_1.extract_sol(b_param)

    x_opt_val = fixed_point_cvxpy(n, A, b_wc, l, u)

    f_star = 0.5 * np.sum((A @ x_star_val - b_wc) ** 2)
    f_opt = 0.5 * np.sum((A @ x_opt_val - b_wc) ** 2)
    assert abs(f_star - f_opt) <= 1e-6
    # Also verify the actual optimizer vectors coincide
    assert np.max(np.abs(x_star_val - x_opt_val)) <= 1e-6
