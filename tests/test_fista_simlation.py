import numpy as np

from fomtools.verifier import Verifier


def test_fista_simlator():
    """
    Tests that the FISTA algorithm without errors.

    Note
    ----
        This test verifies that the simulator runs successfully from an execution
        standpoint. It ensures that the length of the returned output matches the
        input sample and that the results converge. However, it does not assess the
        correctness or accuracy of the simulation results.
    """
    np.random.seed(0)

    p, n, lambd = 3, 4, 1e-1
    D_value = np.random.normal(size=(p, n))
    D_mask = np.random.binomial(n=1, p=0.2, size=(p, n))
    D = np.multiply(D_value, D_mask)


    L = np.real(np.max(np.linalg.eigvals(D.T @ D)))
    eta = 1 / L
    K = 10
    n_samples = 10000



    num_star = 20
    z_star_value = np.random.normal(size=(n, num_star))
    z_star_mask = np.random.binomial(n=1, p=0.1, size=(n, num_star))
    z_star = np.multiply(z_star_value, z_star_mask)

    # noise err
    err = np.random.normal(scale=1e-2, size=(p, num_star))

    # parameter x = D z_star + err
    x = D @ z_star + err

    # parameter set X = [x_lb, x_ub]
    x_lb, x_ub = np.min(x, axis=1), np.max(x, axis=1)

    _rng = np.random.default_rng(0)
    vp = Verifier(method='simulator', obbt_handler=None)

    x_samples = _rng.uniform(low=x_lb, high=x_ub, size=(n_samples, p))
    x_param = vp.add_param(ndim=p, samples=x_samples)
    s = vp.add_iterate(ndim=n, init=np.zeros(n))
    _t = 1 #momentum parameter

    def z_new(y_new, y_prev, t, t_new):
        return y_new + ((t-1)/t_new) * (y_new - y_prev)

    def t_new(t):
        return (1.0 + np.sqrt(1.0 + 4.0 * t**2)) / 2.0

    vp.step.soft_threshold(s - eta * D.T @ (D @ s - x_param), lambd=lambd*eta)
    residuals = vp.solve(K=K, momentum_fn=z_new, step_fn=t_new, momentum_param=_t)
    assert vp['n_samples'] == len(x_samples)
    assert vp['risk_bound'] >= vp['empirical_risk']
    assert residuals[:, -1].mean() < residuals[:, 0].mean()

    print(
        f"Summary statistics:\n"
        f"mean: {vp['mean']:.6e}\n"
        f"std: {vp['std']:.6e}\n"
        f"min: {vp['min']:.6e}\n"
        f"max: {vp['max']:.6e}\n"
        f"median: {vp['median']:.6e}\n"
        f"empirical_risk: {vp['empirical_risk']:.4f}\n"
        f"risk_bound: {vp['risk_bound']:.4f}\n"
        f"n_samples: {vp['n_samples']}\n"
        f"n_iteration: {vp['n_iterations']}\n"
        f"runtime: {vp['runtime']:.4f}\n"
        f"time_per_sample: {vp['time_per_sample']:.4f}\n"
        f"throughput: {vp['throughput']:.4f}\n"
        f"residuals: {residuals}"
    )
