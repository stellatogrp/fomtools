"""
Halpern iteration verification — proof of concept / experimental.

These tests verify the MILP encoding of Halpern iteration with and without
dynamic restart. The implementation is experimental, it uses one hot encoding 
with many many binary variables. 

"""
import numpy as np
import pytest

from fomtools.verifier import Verifier

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)


@pytest.fixture
def vp_halpern():
    return Verifier(solver="gurobi", obbt_handler="gurobi", params={"OutputFlag": 0})


def test_halpern_dynamic(vp_halpern):
    """
    Compare VP worst-case residuals vs Monte Carlo sampling residuals for
    Halpern iteration with dynamic restart.
    Operator: T(x) = ReLU(x - gamma * (P*x + param)).
    """
    VP = vp_halpern
    n = 3
    K = 3
    EPS = np.e
    BIG_M = 200.0

    rng = np.random.default_rng(seed=42)
    Q = rng.standard_normal((n, n))
    P = Q.T @ Q
    L_P = np.linalg.eigvalsh(P).max()
    gamma = 1.0 / L_P

    x_lb, x_ub = -1.5, 1.5

    x_param = VP.add_param(n, lb=x_lb, ub=x_ub)

    z = [None] * (K + 1)
    a = [None] * (K + 1)

    z[0] = VP.add_iterate(n, lb=1, ub=1)
    a[0] = VP.add_iterate(n, lb=1, ub=1)

    #T(x) = ReLU(x - gamma * (P*x + param))
    def operator_func(x_expr):
        grad = P @ x_expr + x_param
        return VP.step.relu(rhs=x_expr - gamma * grad)

    wc_residuals = []

    for k in range(K):
        z[k + 1], a[k + 1] = VP.halpern_step(
            current_step=z[k],
            anchor=a[k],
            operator_func=operator_func,
            K=K,
            restart=True,
            big_M=BIG_M,
        )
        VP.set_infinity_norm_objective(z[k + 1] - z[k])
        res = VP.solve()
        wc_residuals.append(res)

    # comparison baseline
    alpha_m = np.array([(m + 1) / (m + 2) for m in range(K + 1)])
    N = 300
    sample_residuals = np.zeros((N, K))

    for s in range(N):
        x_sample = rng.uniform(x_lb, x_ub, n)
        z_sample = np.ones(n)
        a_sample = z_sample.copy()
        m_counter = 0

        for k in range(K):
            grad_sample = P @ z_sample + x_sample
            t_sample = np.maximum(z_sample - gamma * grad_sample, 0.0)
            rho_val = np.max(np.abs(z_sample - t_sample))

            grad_anchor = P @ a_sample + x_sample
            t_anchor = np.maximum(a_sample - gamma * grad_anchor, 0.0)
            sigma_val = np.max(np.abs(a_sample - t_anchor))

            if rho_val <= (1.0 / EPS) * sigma_val:
                a_sample = t_sample.copy()
                m_counter = 0
            else:
                m_counter = min(m_counter + 1, K)

            alpha = alpha_m[m_counter]
            z_next = alpha * t_sample + (1 - alpha) * a_sample
            sample_residuals[s, k] = np.max(np.abs(z_next - z_sample))
            z_sample = z_next.copy()

    mc_max = np.max(sample_residuals, axis=0)

    assert all(r is not None for r in wc_residuals)
    assert all(r >= 0 for r in wc_residuals)

    wc = np.asarray(wc_residuals)
    tol = 1e-8
    assert np.all(wc + tol >= mc_max), (
        "VP worst-case residuals must upper-bound Monte Carlo maxima.\n"
        f"  VP: {wc}\n  MC: {mc_max}"
    )


def test_halpern_nondynamic(vp_halpern):
    """
    Verify non-dynamic Halpern iteration (fixed anchor, fixed step sizes).

    With restart=False, a^{k+1} = a^k (constant anchor) and
    z^{k+1} = ((k+1)/(k+2)) * T(z^k) + (1/(k+2)) * a^k.
    """
    VP = vp_halpern
    n = 3
    K = 3
    BIG_M = 100.0

    rng = np.random.default_rng(seed=7)

    Q = rng.standard_normal((n, n))
    P = Q.T @ Q
    L_P = np.linalg.eigvalsh(P).max()
    gamma = 1.0 / L_P

    x_lb, x_ub = -1.0, 1.0

    x_param = VP.add_param(n, lb=x_lb, ub=x_ub)

    z = [None] * (K + 1)
    a = [None] * (K + 1)

    z[0] = VP.add_iterate(n, lb=1, ub=1)
    a[0] = VP.add_iterate(n, lb=1, ub=1)

    def operator_func(x_expr):
        grad = P @ x_expr + x_param
        return VP.step.relu(rhs=x_expr - gamma * grad)

    wc_residuals = []

    for k in range(K):
        z[k + 1], a[k + 1] = VP.halpern_step(
            current_step=z[k],
            anchor=a[k],
            operator_func=operator_func,
            K=K,
            restart=False,
            big_M=BIG_M,
        )
        VP.set_infinity_norm_objective(z[k + 1] - z[k])
        res = VP.solve()
        wc_residuals.append(res)

    #MC baseline   
    N = 200
    sample_residuals = np.zeros((N, K))

    for s in range(N):
        x_sample = rng.uniform(x_lb, x_ub, n)
        z_sample = np.ones(n)
        a_sample = np.ones(n)  # anchor stays fixed

        for k in range(K):
            grad_sample = P @ z_sample + x_sample
            t_sample = np.maximum(z_sample - gamma * grad_sample, 0.0)
            alpha_k = (k + 1) / (k + 2)
            z_next = alpha_k * t_sample + (1 - alpha_k) * a_sample
            sample_residuals[s, k] = np.max(np.abs(z_next - z_sample))
            z_sample = z_next.copy()

    mc_max = np.max(sample_residuals, axis=0)

    assert all(r is not None for r in wc_residuals)
    assert all(r >= 0 for r in wc_residuals)

    wc = np.asarray(wc_residuals)
    tol = 1e-8
    assert np.all(wc + tol >= mc_max), (
        "VP worst-case residuals must upper-bound Monte Carlo maxima (non-dynamic).\n"
        f"  VP: {wc}\n  MC: {mc_max}"
    )
