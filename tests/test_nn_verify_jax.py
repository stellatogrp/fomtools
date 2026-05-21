import numpy as np
import pytest

from fomtools.nn import from_jax
from fomtools.verifier import Verifier

flax = pytest.importorskip("flax")
nnx = pytest.importorskip("flax.nnx")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


class ReLUFFNet(nnx.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, rngs: nnx.Rngs):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nnx.Linear(prev, h, rngs=rngs))
            layers.append(lambda x: nnx.relu(x))
            prev = h
        layers.append(nnx.Linear(prev, output_dim, rngs=rngs))
        self.net = nnx.Sequential(*layers)

    def __call__(self, x):
        return self.net(x)


@pytest.fixture(scope="module")
def small_model():
    return ReLUFFNet(input_dim=2, hidden_dims=[4], output_dim=2, rngs=nnx.Rngs(42))


def test_nn_solve(vp, small_model):
    layers = from_jax(small_model)

    input_expr = vp.add_param(2, lb=-1.0, ub=1.0)
    output_expr = vp.neural_network(input_expr, layers)
    final_expr = vp.step.soft_threshold(output_expr, lambd=0.1)

    vp.set_objective_norm(final_expr, np.inf)
    result = vp.solve()

    assert result is not None
    assert result >= 0


def test_nn_simulator(small_model):
    N = 1000
    rng = np.random.default_rng(0)
    x_samples = rng.uniform(-1.0, 1.0, size=(N, 2)).astype(np.float32)

    jit_model = jax.jit(small_model)
    y_jax = jit_model(jnp.array(x_samples))
    y_numpy = np.array(y_jax)

    vp = Verifier(method="simulator", obbt_handler=None)
    layers = from_jax(small_model)
    input_param = vp.add_param(2, lb=-1.0, ub=1.0, samples=x_samples)
    output_expr = vp.neural_network(input_param, layers)

    vp.set_objective_norm(output_expr, np.inf)
    residuals = vp.solve()

    assert residuals.shape == (N, 1)
    expected = np.max(np.abs(y_numpy), axis=1)
    np.testing.assert_allclose(np.array(residuals[:, 0]), expected, rtol=1e-4, atol=1e-4)
