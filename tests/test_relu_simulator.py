import jax
import numpy as np

from fomtools.verifier import Verifier


def test_relu_simulator():
    """
    Tests that the Simulator code executes without errors.

    Note
    ----
        This test verifies that the simulator runs as intended from an
        execution standpoint, but it does not validate the correctness
        or accuracy of the simulation results.
    """
    K = 10
    n = 3
    step_size = 0.01

    vp = Verifier(method="simulator", obbt_handler=None)

    z = [None] * (K + 1)
    z[0] = vp.add_iterate(ndim=n, init=np.full(n, 1))

    P = np.random.normal(size=(n, n))
    P = P @ P.T

    # q is a problem parameter
    key = jax.random.PRNGKey(0)

    q_samples = jax.random.uniform(key, shape=(10000, n), minval=-1.0, maxval=1.0)
    q_param = vp.add_param(ndim=n, samples=q_samples)

    vp.step.relu(z[0] - step_size * (P @ z[0] + q_param))

    residuals = vp.solve(K=K)
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
        f"n_iteration: {vp['n_iterations']}"
        f"residuals: {residuals}"
    )
