import jax
import jax.numpy as jnp
from scipy import sparse as spa

from fomtools.canonicalizers import Canonicalizer


# todo: add logic to go on gpu (if available)
# todo: compare this with the actual worst case
# todo : after solving this, compute the bounds
class Simulator(Canonicalizer):
    def __init__(self):
        super(Simulator, self).__init__(solver=None)
        self.iterates = []
        self.step = None
        self.step_function = None

    def _sparse_to_jax(self, matrix):
        if spa.issparse(matrix):
            return jnp.array(matrix.toarray())
        return jnp.array(matrix)

    def _linear_expr_to_jax(self, expr):
        step_sum = jnp.array(expr.const)

        for coeff in expr.decomposition_dict.values():
            step_sum = jnp.array([self._sparse_to_jax(coeff) + c for c in step_sum])

        return step_sum

    def add_var(self, iterate):
        self.iterates.append(self._linear_expr_to_jax(iterate))

    def canonicalize_relu(self, step):
        self.step = self._linear_expr_to_jax(step.rhs)
        self.step_function = jax.nn.relu

    def solve_model(self, K):
        @jax.jit
        def _simulate(step):
            """Simulate K steps for a single q sample."""

            z = self.iterates[0]
            residuals = []
            for k in range(K):
                z_new = self.step_function(step)
                residuals.append(jnp.max(jnp.abs(z_new - z)))  # infinity norm
                z = z_new
            return jnp.stack(residuals)

        simulate_batch = jax.jit(jax.vmap(_simulate, in_axes=(0)))
        return simulate_batch(self.step)
