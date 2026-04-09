import numpy as np
from fomtools.verifier import Verifier


def test_affine_simulator():
    """
    Tests that the simulator executes linear
    operation without errors.

    Note
    ----
        This test verifies that the simulator runs successfully from an
        execution standpoint. It ensures that the length of the returned
        output matches the input sample and that the results converge.
        However, it does not assess the correctness or accuracy of the
        simulation results.
    """
    np.random.seed(42)
    K = 10
    n = 3

    X = np.random.normal(size=(n, n))
    P = X.T @ X

    max_eig = np.linalg.eigvalsh(P).max()
    step_size = 1.0 / max_eig

    lhs_mat = np.eye(n) + step_size * P

    vp = Verifier(method="simulator", obbt_handler=None)
    z = [None] * (K + 1)
    z[0] = vp.add_iterate(ndim=n, init=np.full(n, 1))

    rng = np.random.default_rng(0)
    q_samples = rng.uniform(low=-1.0, high=1.0, size=(10000, n))
    q_param = vp.add_param(ndim=n, samples=q_samples)

    vp.step.implicit_linear(z[0] - q_param, lhs_mat=lhs_mat)

    residuals = vp.solve(K=K)

    assert vp["n_samples"] == len(q_samples)
    assert vp["risk_bound"] >= vp["empirical_risk"]
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
