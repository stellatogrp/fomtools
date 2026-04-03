import time
import typing
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as spa

from fomtools import LinearExpression
from fomtools.canonicalizers import Canonicalizer

if typing.TYPE_CHECKING:
    from fomtools import Step

import jax.experimental.sparse as jsparse


# todo: what to do if samples is missing from a leaf expression
class Simulator(Canonicalizer):
    """
    An interface for simulating a Verification Problem using JAX.

    Attributes
    ----------
    param_to_samples : dict[LinearExpression, jnp.ndarray]
        A dictionary mapping each parameter variable (LinearExpression)
        to its sample values. Each array should have shape (num_samples, dim),
        where dim is the dimensionality of the parameter.

    iterate_to_samples : dict[LinearExpression, jnp.ndarray]
        A dictionary mapping each iterate variable (a LinearExpression) to
        its initial values. Each array should have shape (dim,), representing
        the initial state for that iterate.

    step_function: Callable
        A function that executes the step on the underlying ``step`` object.

    device: Desired device type. Supported values are:
        - ``None``: Use the default list returned by ``jax.devices()``.
        - ``"cpu"``: Restrict selection to CPU devices.
        - ``"gpu"``: Restrict selection to GPU devices.

    Note
    ----
        Calling `step.relu(...)` more than once on the same Simulator instance
        will overwrite the previously stored ReLU step. Only the most recent
        call is retained, and earlier calls are silently discarded.
        This class is currently designed for workflows where each operation
        type is invoked at most once per simulation step.
    """

    def __init__(self, random_seed=0):
        super(Simulator, self).__init__(solver=None)
        self.param_to_samples = {}
        self.iterate_to_samples = {}

        self._seed_counter = random_seed
        self.step_function = None
        self.device = jax.devices("cpu")[0]

    @staticmethod
    def _sparse_to_jax(matrix: np.ndarray | spa.spmatrix) -> jnp.ndarray:
        """
        Convert a NumPy or SciPy sparse matrix to a JAX array.

        Parameters
        ----------
        matrix: np.ndarray | scipy.sparse.spmatrix
            The sparse matrix to be converted.

        Returns
        -------
        jnp.ndarray
             A JAX array containing the same data as the input matrix. Sparse
            inputs are converted to their dense representation before being
            returned as a JAX array.
        """
        if spa.issparse(matrix):
            return jsparse.BCOO.from_scipy_sparse(matrix)
        return jnp.array(matrix)

    def _linear_expr_to_jax(
        self, expr: LinearExpression
    ) -> dict[str, list[tuple[LinearExpression, jnp.ndarray]]]:
        """
        Converts a linear expression to a JAX array.

        This function decomposes the provided LinearExpression into its
        constituent coefficients and converts each coefficient to a JAX
        array. The terms of the expression are grouped according to whether
        their corresponding leaf variable belongs to the iterate set or
        the parameter set.

        Parameters
        ----------
        expr: LinearExpression
            The linear expression to be converted.

        Returns
        -------
        dict[str, list[tuple[LinearExpression, jnp.ndarray]]]
            A dictionary mapping term categories (e.g., iterate or parameter
            terms) to lists of tuples (leaf, coeff_jax), where leaf identifies
            the variable and coeff_jax is the corresponding coefficient
            represented as a JAX array

        """
        iterate_coeffs = []
        param_coeffs = []

        for leaf, coeff in expr.decomposition_dict.items():
            coeff_jax = self._sparse_to_jax(coeff)

            if leaf in self.iterate_to_samples:
                iterate_coeffs.append((leaf, coeff_jax))
            elif leaf in self.param_to_samples:
                param_coeffs.append((leaf, coeff_jax))

        return {
            "iterate_coeffs": iterate_coeffs,
            "param_coeffs": param_coeffs,
            "const": jnp.array(expr.const),
        }

    def _next_seed(self) -> int:
        """Generate next random seed for reproducible sampling."""
        seed = self._seed_counter
        self._seed_counter += 1
        return seed

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

        def _register_var(
            _var: LinearExpression,
            var_type: str,
            random_seed: int = 0,
            samples: np.ndarray = None,
            init: np.ndarray = None,
        ) -> None:
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

            random_seed: int, optional
                Random seed for reproducible sampling.

            samples: np.ndarray, optional
                For parameter variables, the sample values of the variable.
                Should have shape (num_samples, dim). Required if the variable is
                a leaf. Ignored for iterate variables.

            init: np.ndarray
                For iterate variables, the initial value. Should have shape (dim,).
                Required if ``var_type`` is ``"iterate"``. Ignored for parameter variables.

            Returns
            -------
            None

            Raises
            ------
            ValueError
                If ``var_type`` is ``"iterate"`` and ``init`` is not provided.

            ValueError
                If ``var_type`` is not one of ``"iterate"`` or ``"param"``.

            ValueError
                If neither samples nor appropriate bounds are provided.

            """
            lb, ub = _var.lb, _var.ub
            key = jax.random.PRNGKey(random_seed)

            has_bounds = np.all(np.isfinite(lb)) and np.all(np.isfinite(ub))
            has_samples = samples is not None or init is not None

            if has_bounds and has_samples:
                warnings.warn(
                    f"Bounds and samples both are provided for the {var_type}, bounds will be ignored."
                )

            if var_type == "iterate":
                if init is None:
                    if has_bounds:
                        init = jax.random.uniform(
                            key, shape=(_var.ndim,), minval=lb, maxval=ub
                        )  # pick random values within the bounds
                    else:
                        raise ValueError(
                            "An initial value or finite bounds must be provided when registering an iterate variable."
                        )
                self.iterate_to_samples[_var] = jnp.array(init, dtype=jnp.float32)

            elif var_type == "param":
                if samples is not None:
                    self.param_to_samples[_var] = jnp.array(samples)

            else:
                raise ValueError("Unknown variable type")

        try:
            _register_var(var, random_seed=self._next_seed(), **kwargs)
        except:
            pass

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

    def _prepare_simulation_data(self):
        """
        Prepare stacked arrays and slices for simulation.

        This function packs the iterate initial values and parameter sample
        arrays into contiguous JAX arrays. It also records slice indices that
        map each original variable (``leaf``) to its corresponding location in
        the concatenated arrays.

        Iterate initial values are concatenated into a single vector ``z0``.
        Parameter samples are concatenated along axis 1 into ``q_samples``.
        The returned slice dictionaries allow reconstruction or indexing of
        the individual variables within these packed arrays.

        Returns
        -------
        dict
            A dictionary containing the concatenated simulation inputs and their
            corresponding slice mappings, where:
                - ``z0`` maps to a JAX array containing the concatenation of all
                    iterate initial values.
                - ``p_samples`` maps to a JAX array containing the concatenation (along axis 1)
                    of all parameter sample arrays.
                - ``z_slices`` maps to a dictionary mapping each iterate identifier (leaf)
                    to a slice indicating its location within z0.
                - ``p_slices`` maps to a dictionary mapping each parameter identifier (leaf) to a
                    slice indicating its location within the second dimension of q_samples.”

        """

        def stack_with_slices(
            data_dict: dict[LinearExpression, jnp.ndarray], axis: int = 0
        ) -> tuple[jnp.ndarray, dict[LinearExpression, slice]]:
            """Prepare stacked arrays and slices for simulation."""
            slices = {}
            offset = 0
            for leaf, arr in data_dict.items():
                dim = arr.shape[axis]
                slices[leaf] = slice(offset, offset + dim)
                offset += dim

            if data_dict:
                stacked = jnp.concatenate(list(data_dict.values()), axis=axis)
            else:
                stacked = jnp.array([])

            return stacked, slices

        z0, z_slices = stack_with_slices(self.iterate_to_samples, axis=0)
        p_samples, p_slices = stack_with_slices(self.param_to_samples, axis=1)

        return {
            "z0": z0,
            "p_samples": p_samples,
            "z_slices": z_slices,
            "p_slices": p_slices,
        }

    def _register_statistics(
        self, residuals: jnp.ndarray, delta: float, epsilon: float, total_time: float
    ):
        """
        Compute and register summary statistics from simulation residuals.

        Parameters
        ----------
        residuals : jnp.ndarray
            Shape (N, K) array of residuals at each iteration.

        delta: float
            Confidence parameter $\delta \in (0,1)$. The bound holds with
            probability at least $1 - \delta$.

        epsilon: float
            Tolerance level $\epsilon$ defining the $0$–$1$ error.

        total_time: float
            Total time taken by the simulation.

        """

        final_residuals = residuals[:, -1]
        n_samples = residuals.shape[0]

        self.summary_statistics = {
            "n_samples": n_samples,
            "n_iterations": residuals.shape[1],
            "final_residuals": final_residuals,
            "mean": float(jnp.mean(final_residuals)),
            "std": float(jnp.std(final_residuals)),
            "min": float(jnp.min(final_residuals)),
            "max": float(jnp.max(final_residuals)),
            "median": float(jnp.median(final_residuals)),
            "quantiles": {
                0.25: float(jnp.quantile(final_residuals, 0.25)),
                0.75: float(jnp.quantile(final_residuals, 0.75)),
                0.90: float(jnp.quantile(final_residuals, 0.90)),
                0.95: float(jnp.quantile(final_residuals, 0.95)),
                0.99: float(jnp.quantile(final_residuals, 0.99)),
            },
            "epsilon": epsilon,
            "delta": delta,
            "empirical_risk": float(jnp.mean((final_residuals > epsilon).astype(float))),
            "risk_bound": self.compute_risk_bound(residuals, delta, epsilon),
            "runtime": total_time,
            "time_per_sample": total_time / n_samples,
            "throughput": n_samples / total_time,
        }

    def _to_device(self, arr):
        """Move array to the configured device."""
        return jax.device_put(jnp.array(arr), self.device)

    def _configure_device(self, device: str) -> None:
        """Configure the JAX execution device for the current instance."""
        if device is None:
            devices = jax.devices()
        elif device in {"cpu", "gpu"}:
            devices = jax.devices(device)
        else:
            raise ValueError(f"Unsupported device type: {device}. Expected None, 'cpu', or 'gpu'.")

        if not devices:
            raise RuntimeError(f"No available JAX devices for type: {device}")

        self.device = devices[0]

    def canonicalize_relu(self, step: "Step") -> None:
        """
        Registers the provided ReLU step for JAX simulation.

        This method converts the linear expression defining the input to the
        ReLU operation into a JAX-evaluable form and constructs a corresponding
        update function. The update function has signature ``f(z_stacked, p_stacked)``
        where the iterate and parameter variables are provided as concatenated vectors.
        Slice mappings are used internally to extract the relevant components for
        each variable.

        The constructed function is stored in ``self.step_function`` and can be
        used during simulation to evaluate the ReLU update.

        Parameters
        ----------
        step: Step
            A step object containing the ReLU operation. The `step.rhs` attribute
            holds the LinearExpression representing the input to the ReLU.

        Returns
        -------
        None

        """
        coeffs = self._linear_expr_to_jax(step.rhs)

        iterate_coeffs = coeffs["iterate_coeffs"]
        param_coeffs = coeffs["param_coeffs"]
        const = coeffs["const"]

        simulation_data = self._prepare_simulation_data()

        def update_fn(z_stacked: jnp.ndarray, p_stacked: jnp.ndarray) -> jnp.ndarray:
            """
            Evaluate the linear update followed by a ReLU activation.

            This function reconstructs the value of a linear expression from
            stacked iterate and parameter vectors. The contributions of each
            variable are computed using the precomputed coefficient matrices
            and slice mappings. A ReLU activation is applied to the resulting
            value.

            Parameters
            ----------
            z_stacked: jnp.ndarray
                Concatenation of all iterate variables, with shape. Individual iterate
                values are retrieved using slices stored in ``simulation_data["z_slices"]``.

            p_stacked: jnp.ndarray
                Concatenation of all parameter variables. Individual parameter values
                are retrieved using slices stored in ``simulation_data["p_slices"]``.

            Returns
            -------
            jnp.ndarray
                The value of the linear expression evaluated at the provided
                iterates and parameters, after applying a ReLU activation.
            """
            result = const.copy()

            for leaf, coeff in iterate_coeffs:
                z_slice = z_stacked[simulation_data["z_slices"][leaf]]
                result += coeff @ z_slice

            for leaf, coeff in param_coeffs:
                p_slice = p_stacked[simulation_data["p_slices"][leaf]]
                result += coeff @ p_slice

            return jax.nn.relu(result)

        self.step_function = update_fn

    def solve_model(
        self, K: int, delta: float = 0.05, epsilon: float = 0.0, device: str = None
    ) -> jnp.ndarray:
        """
        Run a batched simulation of the registered step function.

        This method simulates the iterative dynamics defined by
        ``self.step_function`` starting from the initial iterate vector.
        The simulation is performed for each parameter sample.
        For every sample, the step function is applied sequentially for
        ``K`` iterations. At each step, the infinity norm of the state update
        (``max(abs(z_{k+1} - z_k))``) is recorded as a residual.

        After the simulation, a risk bound is computed from the residual
        trajectories using :meth:`compute_risk_bound`.

        Parameters
        ----------
        K: int
            Number of iterations to simulate.

        delta: float, optional
            Confidence parameter $\delta \in (0,1)$. The bound holds with
            probability at least $1 - \delta$.

        epsilon: float, optional
            Tolerance level $\epsilon$ defining the $0$–$1$ error.

        device: str, optional
            The device to run the simulation on. If not specified, the default is `None`,
            and the simulation will run on the host system without specific device
            acceleration. The valid options are:

            - `None`: No device (default behavior, runs on the host).
            - `"cpu"`: Run on the CPU.
            - `"gpu"`: Run on the GPU, if available.

        Returns
        -------
        residuals: jnp.ndarray
           A 2D array of shape ``(num_samples, K)`` containing the residual
           (infinity norm of successive state differences) at each iteration
           for each parameter sample.

        Raises
        ------
        ValueError
            If no step function has been registered with the simulator.
        """
        self._configure_device(device)
        if self.step_function is not None:
            update_fn = self.step_function
        else:
            raise ValueError(
                "No step function has been registered with this "
                "simulator instance. Please register a step function "
                "before running the simulation"
            )
        simulation_data = self._prepare_simulation_data()
        z0 = self._to_device(simulation_data["z0"])
        p_samples = self._to_device(simulation_data["p_samples"])

        @jax.jit
        def _simulate(samples):
            """Simulate K steps for a single sample."""

            def body(z, _):
                z_new = update_fn(z, samples)
                residual = jnp.max(jnp.abs(z_new - z))  # infinity norm
                return z_new, residual

            _, residuals = jax.lax.scan(body, z0, None, length=K)
            return residuals

        simulate_batch = jax.vmap(_simulate, in_axes=0)

        # warm-up: triggers compilation, discard result
        simulate_batch(p_samples).block_until_ready()

        # timing pure execution
        start = time.perf_counter()
        residuals = simulate_batch(p_samples).block_until_ready()
        elapsed = time.perf_counter() - start

        self._register_statistics(residuals, delta, epsilon, elapsed)
        return residuals
