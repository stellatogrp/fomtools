import numpy as np
import scipy.sparse as spa

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

    output_mat = np.eye(n) + step_size * P

    vp = Verifier(method="simulator", obbt_solver=None)
    z = [None] * (K + 1)
    z[0] = vp.iterate(ndim=n, init=np.full(n, 1))

    rng = np.random.default_rng(0)
    q_samples = rng.uniform(low=-1.0, high=1.0, size=(10000, n))
    q_param = vp.parameter(ndim=n, samples=q_samples)

    for k in range(1, K + 1):
        z[k] = vp.implicit_linear(z[k - 1] - q_param, output_mat=output_mat)

    residuals = vp.solve()

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


def test_affine_sparse_simulator():
    """
    Tests that the simulator executes the affine (implicit linear) step
    correctly using a sparse tridiagonal output_mat.

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

    diagonals = [-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)]
    A = spa.diags(diagonals, offsets=[-1, 0, 1], format="csr")

    max_eig = 4.0
    step_size = 1.0 / max_eig

    output_mat = spa.eye(n, format="csr") + step_size * A

    vp = Verifier(method="simulator", obbt_solver=None)
    z = [None] * (K + 1)
    z[0] = vp.iterate(ndim=n, init=np.full(n, 1))

    rng = np.random.default_rng(0)
    q_samples = rng.uniform(low=-1.0, high=1.0, size=(10000, n))
    q_param = vp.parameter(ndim=n, samples=q_samples)

    for k in range(1, K + 1):
        z[k] = vp.implicit_linear(z[k - 1] - step_size * q_param, output_mat=output_mat)

    residuals = vp.solve()

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
