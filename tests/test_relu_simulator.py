import jax
import jax.numpy as jnp
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
    q_offset = 1.0

    vp = Verifier(method="simulator", obbt_handler=None)

    z = [None] * (K + 1)
    z[0] = vp.add_iterate(ndim=n, lb=-1.0, ub=1.0)

    q_nominal = np.random.normal(size=n)
    P = np.random.normal(size=(n, n))
    P = P @ P.T

    # q is a problem parameter
    q_param = vp.add_param(ndim=n, lb=q_nominal - q_offset, ub=q_nominal + q_offset)

    key = jax.random.PRNGKey(0)
    q_samples = jax.random.uniform(key, shape=(10000, n), minval=-1.0, maxval=1.0)
    vp.step.relu(z[0] - step_size * (P @ z[0] + q_param))
    residuals, upper_bound = vp.solve(K=K, z0=jnp.full(n, -1), q_samples=q_samples)
