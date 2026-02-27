import typing

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as spa

from fomtools import LinearExpression
from fomtools.canonicalizers import Canonicalizer

if typing.TYPE_CHECKING:
    from fomtools import Step


# todo: add logic to go on gpu (if available)
# todo: compare this with the actual worst case
class Simulator(Canonicalizer):
    """
    An interface for simulating a Verification Problem using JAX.

    Attributes
    ----------
    iterates : List[jnp.ndarray]
        A list of iterates stored as ``jnp.ndarray`` instances that are
        registered with the Simulator instance.

    expression : jnp.ndarray
        Current parameter array passed to the step function.

    step_function: Callable
        A function that executes the step on the underlying
            ``step`` object.

    """

    def __init__(self):
        super(Simulator, self).__init__(solver=None)
        self.iterates = set()
        self.params = set()
        self.expression = None
        self.step_function = None

    @staticmethod
    def _sparse_to_jax(matrix: np.ndarray | spa.spmatrix) -> jnp.ndarray:
        """
        Convert a NumPy or SciPy sparse matrix to a JAX array.

        Parameters
        ----------
        matrix : np.ndarray | scipy.sparse.spmatrix
            The sparse matrix to be converted.

        Returns
        -------
        jnp.ndarray
             A JAX array containing the same data as the input matrix. Sparse
            inputs are converted to their dense representation before being
            returned as a JAX array.
        """
        if spa.issparse(matrix):
            return jnp.array(matrix.toarray())
        return jnp.array(matrix)

    def _linear_expr_to_jax(self, expr: LinearExpression) -> jnp.ndarray:
        """
        Converts a linear expression to a JAX array.

        Parameters
        ----------
        expr : LinearExpression
            The linear expression to be converted.

        Returns
        -------
        jnp.ndarray
            A JAX array containing the same information as the provided
            LinearExpression object.
        """
        ndim = expr.ndim
        z_coeff = jnp.zeros((ndim, ndim))
        q_coeff = jnp.zeros((ndim, ndim))

        for leaf, coeff in expr.decomposition_dict.items():
            coeff_jax = self._sparse_to_jax(coeff)

            if leaf in self.iterates:
                z_coeff += coeff_jax
            elif leaf in self.params:
                q_coeff += coeff_jax
            else:
                raise ValueError(
                    f"Variable {leaf} has not been registered with this "
                    "simulator instance. Please register a variable "
                    "before running the simulation"
                )

        return {"z_coeff": z_coeff, "q_coeff": q_coeff, "const": jnp.array(expr.const)}

    def add_var(self, var: LinearExpression, **kwargs) -> None:
        """
        Add a variable to the Simulator instance. This method converts the provided
        LinearExpression object to a JAX array and registers it with the Simulator
        instance so that it can be referenced later.

        Parameters
        ----------
        var: LinearExpression
            Linear expression object to add to the Simulator instance.

        Returns
        -------
        None
        """

        def _register_var(_var: LinearExpression, var_type: str) -> None:
            """
            Register a variable to the Simulator instance.

            This method adds the given ``LinearExpression`` object to the appropriate
            internal collection based on its role in the simulation.

            Parameters
            ----------
            _var: LinearExpression
                Linear expression object to add to the Simulator instance.

            var_type : str
                The category of the variable. Must be one of:
                    - "iterate": registers the variable as an iterate.
                    - "param": registers the variable as a parameter.

            Returns
            -------
            None
            """
            if var_type == "iterate":
                self.iterates.add(_var)
            elif var_type == "param":
                self.params.add(_var)
            else:
                raise ValueError(f"Unknown var_type: {var_type}")

        _register_var(var, **kwargs)

    @staticmethod
    def compute_risk_bound(residuals: jnp.ndarray, delta: float = 0.05, epsilon: float = 0.0):
        r"""
        Compute a high-probability bound on the risk using the KL inverse bound.

        Given a tolerance $\epsilon$, a number of iterations $k$ (implicitly given by the last
        column of ``residuals``),  and $N$ sampled problem instances,

        we define the $0$–$1$ error for sample $x$ as

        $$
            e(x) = \mathbf{1}\{ \phi(z^k(x), x) > \epsilon \},
        $$

        which in this implementation corresponds to checking whether the
        final residual exceeds $\epsilon$.

        Let $N$ be the number of samples and let the empirical risk be


        $$
            \hat{r}_S
            =
            \frac{1}{N}
            \sum_{i=1}^{N}
            \mathbf{1}\{ r_i^{(k)} > \epsilon \},
        $$

        where $r_i^{(k)}$ is the final residual of trajectory $i$.

        Then, with probability at least $1 - \delta$, the true risk $r$
        satisfies

        $$
            r
            \;\le\;
            \mathrm{kl}^{-1}
            \!\left(
                \hat{r}_S
                \;\middle|\;
                \frac{\log(2/\delta)}{N}
            \right),
        $$

        where $\mathrm{kl}^{-1}(q \mid c)$ denotes the largest $p \in [q,1]$
        such that

        $$
            \mathrm{KL}(q \,\|\, p)
            \le c,
        $$

        and

        $$
            \mathrm{KL}(q \,\|\, p)
            =
            q \log\!\frac{q}{p}
            +
            (1-q)\log\!\frac{1-q}{1-p}
        $$

        is the binary Kullback–Leibler divergence.

        Parameters
        ----------
        residuals: jnp.ndarray
            Residuals of the problem instances.

        delta: float, optional
            Confidence parameter $\delta \in (0,1)$. The bound holds with
            probability at least $1 - \delta$.

        epsilon: float, optional
            Tolerance level $\epsilon$ defining the $0$–$1$ error.

        Returns
        -------
        float
            Upper bound on true risk via KL inverse.

        """

        def _kl_inverse(q, c):
            """
            Compute kl⁻¹(q | c): largest p such that KL(q || p) ≤ c.
            """
            if q >= 1.0:
                return 1.0
            if c == 0:
                return q

            low, high = q, 1.0
            for _ in range(100):
                mid = (low + high) / 2
                if mid == 1.0:
                    kl = float("inf")
                else:
                    kl = q * np.log(q / mid + 1e-10) + (1 - q) * np.log((1 - q) / (1 - mid) + 1e-10)

                if kl < c:
                    low = mid
                else:
                    high = mid
            return high

        N = residuals.shape[0]
        final_residuals = residuals[:, -1]

        errors = (final_residuals > epsilon).astype(float)
        empirical_risk = jnp.mean(errors)

        c = jnp.log(2 / delta) / N
        return _kl_inverse(float(empirical_risk), float(c))

    def canonicalize_relu(self, step: "Step") -> None:
        """
        Registers the provided ReLU step for JAX simulation.

        This function extracts the linear expression from the step's
        right-hand side (rhs), converts it to JAX-compatible coefficient
        matrices, and stores the corresponding JAX numerical operation.

        Parameters
        ----------
        step : Step
            A step object containing the ReLU operation. The `step.rhs` attribute
            holds the LinearExpression representing the input to the ReLU.

        Returns
        -------
        None

        """
        expression = self._linear_expr_to_jax(step.rhs)
        z_coeff = expression["z_coeff"]
        q_coeff = expression["q_coeff"]
        const = expression["const"]

        def update_fn(z, q):
            return jax.nn.relu(z_coeff @ z + q_coeff @ q + const)

        self.step_function = update_fn

    def solve_model(self, K, z0, q_samples, delta=0.05, epsilon=0.0):
        # todo: need to polish docstring here
        """
        Run a batched simulation of the step function over K iterations.

        This method evaluates the simulator dynamics starting from an initial
        state ``z0`` for each parameter sample in ``q_samples``. For every sample,
        the registered ``step_function`` is applied iteratively ``K`` times.
        At each step, the infinity norm of the state update
        (``max(abs(z_{k+1} - z_k))``) is recorded as a residual.

        Parameters
        ----------
        K : int
            Number of iterations to simulate.

        z0 : jnp.ndarray
            Initial state of the system.

        q_samples : jnp.ndarray
            Batch of parameter samples. The first dimension is interpreted
            as the batch dimension and is vectorized over.

        Returns
        -------
        jax.numpy.ndarray
            A 2D array of shape ``(num_samples, K)`` containing the residual
            (infinity norm of successive state differences) at each iteration
            for each parameter sample.

        Raises
        ------
        ValueError
            If no step function has been registered with the simulator.
        """
        if self.step_function is not None:
            update_fn = self.step_function
        else:
            raise ValueError(
                "No step function has not been registered with this "
                "simulator instance. Please register a step function "
                "before running the simulation"
            )

        z0_jax = z0
        q_samples_jax = jnp.array(q_samples)

        @jax.jit
        def _simulate(samples):
            """Simulate K steps for a single q sample."""
            z = z0_jax
            residuals = []
            for k in range(K):
                z_new = update_fn(z, samples)
                residuals.append(jnp.max(jnp.abs(z_new - z)))  # infinity norm
                z = z_new
            return jnp.stack(residuals)

        simulate_batch = jax.jit(jax.vmap(_simulate, in_axes=0))
        residuals = simulate_batch(q_samples_jax)
        bounds = self.compute_risk_bound(residuals, delta=delta, epsilon=epsilon)

        return residuals, bounds
