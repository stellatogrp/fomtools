import time
import typing
import warnings

try:
    import cupy as cp
    import cupyx.scipy.sparse as csp

    # Force CUDA initialization to happen NOW so we catch the error here
    cp.cuda.runtime.getDeviceCount()

    CUDA_AVAILABLE = True
except ImportError:
    import numpy as cp  # fallback to numpy
    import scipy.sparse as csp

    CUDA_AVAILABLE = False


import numpy as np
import scipy.sparse as spa

from fomtools import LinearExpression
from fomtools.canonicalizers import Canonicalizer

if typing.TYPE_CHECKING:
    from fomtools import Step

from contextlib import nullcontext
from typing import Any


# todo: what to do if samples is missing from a leaf expression
class Simulator(Canonicalizer):
    """
    An interface for simulating a Verification Problem using CuPy.

    Attributes
    ----------
    param_to_samples : dict[LinearExpression, cp.ndarray]
        A dictionary mapping each parameter variable (LinearExpression)
        to its sample values. Each array should have shape (num_samples, dim),
        where dim is the dimensionality of the parameter.

    iterate_to_samples : dict[LinearExpression, cp.ndarray]
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
        self.device = 0  # GPU device ID

    @staticmethod
    def _sync():
        """Synchronize the CUDA stream if running on GPU, otherwise no-op."""
        if CUDA_AVAILABLE:
            cp.cuda.Stream.null.synchronize()

    @staticmethod
    def _get_device_context(device):
        """Return the active CUDA device context, or a no-op context on CPU."""
        return cp.cuda.Device(device) if CUDA_AVAILABLE else nullcontext()  # no-op on CPU

    @staticmethod
    def _sparse_to_cp(matrix: np.ndarray | spa.spmatrix) -> cp.ndarray | csp.csr_matrix:
        """
        Convert a NumPy or SciPy sparse matrix to a CuPY array
        or cupyx CSR sparse matrix, respectively.

        Parameters
        ----------
        matrix: np.ndarray | scipy.sparse.spmatrix
            The sparse matrix to be converted.

        Returns
        -------
        cp.ndarray | cupyx.scipy.sparse.csr_matrix
             A CuPy array containing the same data as the input matrix.
        """
        if spa.issparse(matrix):
            return csp.csr_matrix(matrix)
        return cp.array(matrix)

    def _linear_expr_to_cupy(
        self, expr: LinearExpression
    ) -> dict[str, list[tuple[LinearExpression, cp.ndarray | csp.csr_matrix]]]:
        """
        Converts a linear expression to CuPy arrays (dense) or cupyx CSR
        matrices (sparse).

        This function decomposes the provided LinearExpression into its
        constituent coefficients and converts each coefficient to CuPy arrays
        or cupyx CSR matrices. The terms of the expression are grouped
        according to whether their corresponding leaf variable belongs
        to the iterate set or the parameter set.

        Parameters
        ----------
        expr: LinearExpression
            The linear expression to be converted.

        Returns
        -------
        dict[str, list[tuple[LinearExpression, cp.ndarray | csp.csr_matrix]]]
            A dictionary mapping term categories (e.g., iterate or parameter
            terms) to lists of tuples (leaf, coeff_jax), where leaf identifies
            the variable and coeff_cp is the corresponding coefficient
            represented as CuPy arrays or cupyx CSR matrices.

        """
        iterate_coeffs = []
        param_coeffs = []

        for leaf, coeff in expr.decomposition_dict.items():
            coeff_cp = self._sparse_to_cp(coeff)

            if leaf in self.iterate_to_samples:
                iterate_coeffs.append((leaf, coeff_cp))
            elif leaf in self.param_to_samples:
                param_coeffs.append((leaf, coeff_cp))

        return {
            "iterate_coeffs": iterate_coeffs,
            "param_coeffs": param_coeffs,
            "const": cp.array(expr.const, dtype=cp.float32),
        }

    def _next_seed(self) -> int:
        """Generate next random seed for reproducible sampling."""
        seed = self._seed_counter
        self._seed_counter += 1
        return seed

    def add_var(self, var: LinearExpression, **kwargs) -> None:
        """
        Add a variable to the Simulator instance. This method converts the provided
        LinearExpression object to a CuPy array and registers it with the Simulator
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
            rng = np.random.default_rng(random_seed)

            has_bounds = np.all(np.isfinite(lb)) and np.all(np.isfinite(ub))
            has_samples = samples is not None or init is not None

            if has_bounds and has_samples:
                warnings.warn(
                    f"Bounds and samples both are provided for the {var_type}, bounds will be ignored."
                )

            if var_type == "iterate":
                if init is None:
                    if has_bounds:
                        init = rng.uniform(lb, ub, size=(_var.ndim,)).astype(np.float32)
                    else:
                        raise ValueError(
                            "An initial value or finite bounds must be provided when registering an iterate variable."
                        )
                self.iterate_to_samples[_var] = cp.array(init, dtype=cp.float32)

            elif var_type == "param":
                if samples is not None:
                    self.param_to_samples[_var] = cp.array(samples, dtype=cp.float32)

            else:
                raise ValueError("Unknown variable type")

        _register_var(var, random_seed=self._next_seed(), **kwargs)

    @staticmethod
    def compute_risk_bound(residuals: cp.ndarray, delta: float = 0.05, epsilon: float = 0.0):
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
        residuals: cp.ndarray
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

        errors = (final_residuals > epsilon).astype(cp.float32)
        empirical_risk = cp.mean(errors)

        c = cp.log(2 / delta) / N
        return _kl_inverse(float(empirical_risk), float(c))

    def _prepare_simulation_data(self):
        """
        Prepare stacked arrays and slices for simulation.

        This function packs the iterate initial values and parameter sample
        arrays into contiguous CuPy arrays. It also records slice indices that
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
                - ``z0`` maps to a CuPy array containing the concatenation of all
                    iterate initial values.
                - ``p_samples`` maps to a CuPy array containing the concatenation (along axis 1)
                    of all parameter sample arrays.
                - ``z_slices`` maps to a dictionary mapping each iterate identifier (leaf)
                    to a slice indicating its location within z0.
                - ``p_slices`` maps to a dictionary mapping each parameter identifier (leaf) to a
                    slice indicating its location within the second dimension of q_samples.”

        """

        def stack_with_slices(
            data_dict: dict[LinearExpression, cp.ndarray], axis: int = 0
        ) -> tuple[cp.ndarray, dict[LinearExpression, slice]]:
            """Prepare stacked arrays and slices for simulation."""
            slices = {}
            offset = 0
            for leaf, arr in data_dict.items():
                dim = arr.shape[axis]
                slices[leaf] = slice(offset, offset + dim)
                offset += dim

            if data_dict:
                stacked = cp.concatenate(list(data_dict.values()), axis=axis)
            else:
                stacked = cp.array([])

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
        self, residuals: cp.ndarray, delta: float, epsilon: float, total_time: float
    ):
        """
        Compute and register summary statistics from simulation residuals.

        Parameters
        ----------
        residuals : cp.ndarray
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
            "mean": float(cp.mean(final_residuals)),
            "std": float(cp.std(final_residuals)),
            "min": float(cp.min(final_residuals)),
            "max": float(cp.max(final_residuals)),
            "median": float(cp.median(final_residuals)),
            "quantiles": {
                0.25: float(cp.quantile(final_residuals, 0.25)),
                0.75: float(cp.quantile(final_residuals, 0.75)),
                0.90: float(cp.quantile(final_residuals, 0.90)),
                0.95: float(cp.quantile(final_residuals, 0.95)),
                0.99: float(cp.quantile(final_residuals, 0.99)),
            },
            "epsilon": epsilon,
            "delta": delta,
            "empirical_risk": float(cp.mean((final_residuals > epsilon).astype(cp.float32))),
            "risk_bound": self.compute_risk_bound(residuals, delta, epsilon),
            "runtime": total_time,
            "time_per_sample": total_time / n_samples,
            "throughput": n_samples / total_time,
        }

    def _to_device(self, arr: np.ndarray | cp.ndarray) -> cp.ndarray | np.ndarray:
        """Move array to the configured device."""
        if CUDA_AVAILABLE:
            with self._get_device_context(self.device):
                return cp.asarray(arr)
        return arr

    def _configure_device(self, device: str) -> None:
        """Configure the CuPy GPU device ID for this instance."""
        if device is None or device == "gpu":
            self.device = 0
        elif device == "cpu":
            warnings.warn(
                "CuPy only supports GPU execution. Defaulting to GPU device 0. "
                "Use NumPy/SciPy directly for CPU-only simulation."
            )
            self.device = 0
        else:
            raise ValueError(f"Unsupported device type: {device}. Expected None, 'cpu', or 'gpu'.")

    @staticmethod
    def _matvec_batch(coeff: cp.ndarray | csp.csr_matrix, x_batch: cp.ndarray) -> cp.ndarray:
        """
        Apply coefficient matrix to a batch of vectors.

        Parameters
        ----------
        coeff: (out_dim, d) # sparse or dense
        x_batch: (N, d) # dense CuPy array

        Returns
        -------
        (N, out_dim) dense CuPy array.
        """
        if csp.issparse(coeff):
            return (coeff @ x_batch.T).T
        return x_batch @ coeff.T

    def _assemble_linear_combination(
        self,
        z_stacked: cp.ndarray,
        p_stacked: cp.ndarray,
        coeffs: dict[str, list[tuple[LinearExpression, cp.ndarray | csp.csr_matrix]]],
        simulation_data: dict[str, Any],
    ) -> cp.ndarray:
        """
        Assemble a batched linear combination of state and parameter slices
        using precomputed coefficients.

        Parameters
        ----------
        z_stacked: cp.ndarray
            Concatenation of all iterate variables, with shape. Individual iterate
            values are retrieved using slices stored in ``simulation_data["z_slices"]``.

        p_stacked: cp.ndarray
            Concatenation of all parameter variables. Individual parameter values
            are retrieved using slices stored in ``simulation_data["p_slices"]``.

        coeffs: dict[str, list[tuple[LinearExpression, cp.ndarray | csp.csr_matrix]]]
            A dictionary mapping term categories (e.g., iterate or parameter
            terms) to lists of tuples (leaf, coeff_jax), where leaf identifies
            the variable and coeff_cp is the corresponding coefficient
            represented as CuPy arrays or cupyx CSR matrices.

        simulation_data: dict[str, Any]
             A dictionary containing the concatenated simulation inputs and their
            corresponding slice mappings, where:
                - ``z0`` maps to a CuPy array containing the concatenation of all
                    iterate initial values.
                - ``p_samples`` maps to a CuPy array containing the concatenation (along axis 1)
                    of all parameter sample arrays.
                - ``z_slices`` maps to a dictionary mapping each iterate identifier (leaf)
                    to a slice indicating its location within z0.
                - ``p_slices`` maps to a dictionary mapping each parameter identifier (leaf) to a
                    slice indicating its location within the second dimension of q_samples.”
        Returns
        -------
        cp.ndarray
            The value of the linear expression evaluated at the provided
            iterates and parameters, after applying a ReLU activation.

        """
        N = z_stacked.shape[0]
        result = cp.tile(coeffs["const"], (N, 1))

        for leaf, coeff in coeffs["iterate_coeffs"]:
            z_slice = z_stacked[:, simulation_data["z_slices"][leaf]]
            result += self._matvec_batch(coeff, z_slice)

        for leaf, coeff in coeffs["param_coeffs"]:
            p_slice = p_stacked[:, simulation_data["p_slices"][leaf]]
            result += self._matvec_batch(coeff, p_slice)

        return result

    def canonicalize_relu(self, step: "Step") -> None:
        """
        Registers the provided ReLU step for CuPy simulation.

        This method converts the linear expression defining the input to the
        ReLU operation into a CuPy-evaluable form and constructs a corresponding
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
        coeffs = self._linear_expr_to_cupy(step.rhs)
        simulation_data = self._prepare_simulation_data()

        def update_fn(z_stacked: cp.ndarray, p_stacked: cp.ndarray) -> cp.ndarray:
            """
            Evaluate the linear update followed by a ReLU activation.

            This function reconstructs the value of a linear expression from
            stacked iterate and parameter vectors. The contributions of each
            variable are computed using the precomputed coefficient matrices
            and slice mappings. A ReLU activation is applied to the resulting
            value.

            Parameters
            ----------
            z_stacked: cp.ndarray
                Concatenation of all iterate variables, with shape. Individual iterate
                values are retrieved using slices stored in ``simulation_data["z_slices"]``.

            p_stacked: cp.ndarray
                Concatenation of all parameter variables. Individual parameter values
                are retrieved using slices stored in ``simulation_data["p_slices"]``.

            Returns
            -------
            cp.ndarray
                The value of the linear expression evaluated at the provided
                iterates and parameters, after applying a ReLU activation.
            """
            result = self._assemble_linear_combination(
                z_stacked, p_stacked, coeffs, simulation_data
            )
            return cp.maximum(result, 0)

        self.step_function = update_fn

    def canonicalize_softthreshold(self, step: "Step") -> None:
        """
        Registers the provided soft-threshold step for CuPy simulation.

        This method converts the linear expression defining the input to the
        soft-threshold operation into a CuPy-evaluable form and constructs a
        corresponding update function. The update function has signature
        ``f(z_stacked, p_stacked)`` where the iterate and parameter variables
        are provided as concatenated vectors. Slice mappings are used internally
        to extract the relevant components for each variable.

        The constructed function is stored in ``self.step_function`` and can be
        used during simulation to evaluate the soft-threshold update.

        Parameters
        ----------
        step: Step
            A step object containing the soft-threshold operation. The
            ``step.rhs`` attribute holds the LinearExpression representing
            the input, and ``step.lambd`` holds the regularization strength.

        Returns
        -------
        None
        """
        coeffs = self._linear_expr_to_cupy(step.rhs)
        simulation_data = self._prepare_simulation_data()
        lambd = step.lambd

        def update_fn(z_stacked: cp.ndarray, p_stacked: cp.ndarray) -> cp.ndarray:
            """
            Evaluate the linear update followed by a soft-threshold activation.

            Parameters
            ----------
            z_stacked: cp.ndarray
                Concatenation of all iterate variables. Individual iterate values
                are retrieved using slices stored in ``simulation_data["z_slices"]``.

            p_stacked: cp.ndarray
                Concatenation of all parameter variables. Individual parameter values
                are retrieved using slices stored in ``simulation_data["p_slices"]``.

            Returns
            -------
            cp.ndarray
                The value of the linear expression evaluated at the provided iterates
                and parameters, after applying a soft-threshold activation.

            """
            result = self._assemble_linear_combination(
                z_stacked, p_stacked, coeffs, simulation_data
            )
            return cp.sign(result) * cp.maximum(cp.abs(result) - lambd, 0)

        self.step_function = update_fn

    def canonicalize_saturatedlinear(self, step: "Step") -> None:
        """
        Registers the provided soft-threshold step for CuPy simulation.

        This method converts the linear expression defining the input to the
        saturated linear operation into a CuPy-evaluable form and constructs a
        corresponding update function. The update function has signature
        ``f(z_stacked, p_stacked)`` where the iterate and parameter variables
        are provided as concatenated vectors. Slice mappings are used internally
        to extract the relevant components for each variable.

        The constructed function is stored in ``self.step_function`` and can be
        used during simulation to evaluate the soft-threshold update.

        Parameters
        ----------
        step: Step
            A step object containing the soft-threshold operation. The
            ``step.rhs`` attribute holds the LinearExpression representing
            the input, and ``step.lambd`` holds the regularization strength.

        Returns
        -------
        None
        """
        coeffs = self._linear_expr_to_cupy(step.rhs)
        simulation_data = self._prepare_simulation_data()
        l = step.l
        u = step.u

        def update_fn(z_stacked: cp.ndarray, p_stacked: cp.ndarray) -> cp.ndarray:
            """
            Evaluate the linear update followed by a saturated linear activation.

            Parameters
            ----------
            z_stacked: cp.ndarray
                Concatenation of all iterate variables. Individual iterate values
                are retrieved using slices stored in ``simulation_data["z_slices"]``.

            p_stacked: cp.ndarray
                Concatenation of all parameter variables. Individual parameter values
                are retrieved using slices stored in ``simulation_data["p_slices"]``.

            Returns
            -------
            cp.ndarray
                The value of the linear expression evaluated at the provided iterates
                and parameters, after applying a saturated linear activation.

            """
            result = self._assemble_linear_combination(
                z_stacked, p_stacked, coeffs, simulation_data
            )
            return cp.clip(result, l, u)

        self.step_function = update_fn

    def canonicalize_affine(self, step: "Step") -> None:
        """
        Registers the provided affine step for CuPy simulation.

        This method converts the linear expression defining the input to the
        affine operation into a CuPy-evaluable form and constructs a
        corresponding update function. The update function has signature
        ``f(z_stacked, p_stacked)`` where the iterate and parameter variables
        are provided as concatenated vectors. Slice mappings are used internally
        to extract the relevant components for each variable.

        The constructed function is stored in ``self.step_functions`` and can be
        used during simulation to evaluate the affine step.

        Parameters
        ----------
        step: Step
            A step object that represents the affine transformation. The `step.rhs`
            attribute contains the `LinearExpression` defining the input, while
            `step.lhs_mat` stores the matrix on the left-hand side of the
            linear operation.

        Returns
        -------
        None
        """
        lhs = self._sparse_to_cp(step.lhs_mat)
        coeffs = self._linear_expr_to_cupy(step.rhs)
        simulation_data = self._prepare_simulation_data()

        # precompute inverse to avoid a solve at every iteration
        lhs_inv = cp.linalg.solve(lhs, cp.eye(lhs.shape[0]))

        def update_fn(z_stacked: cp.ndarray, p_stacked: cp.ndarray) -> cp.ndarray:
            """
            Evaluate the affine update w = L_z @ v + L_p @ x.

            Parameters
            ----------
            z_stacked: cp.ndarray
                Concatenation of all iterate variables. Individual iterate values
                are retrieved using slices stored in ``simulation_data["z_slices"]``.

            p_stacked: cp.ndarray
                Concatenation of all parameter variables. Individual parameter values
                are retrieved using slices stored in ``simulation_data["p_slices"]``.

            Returns
            -------
            cp.ndarray
                The value of the linear expression evaluated at the provided iterates
                and parameters.
            """
            result = self._assemble_linear_combination(
                z_stacked, p_stacked, coeffs, simulation_data
            )
            return self._matvec_batch(lhs_inv, result)

        self.step_function = update_fn

    def solve_model(
        self, K: int, delta: float = 0.05, epsilon: float = 0.0, device: str = None
    ) -> cp.ndarray:
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
        residuals: cp.ndarray
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

        with self._get_device_context(self.device):
            z0 = self._to_device(simulation_data["z0"])
            p_samples = self._to_device(simulation_data["p_samples"])

            N = p_samples.shape[0]
            update_fn = self.step_function

            def _simulate_batch(z0: cp.ndarray, p_samples: cp.ndarray) -> cp.ndarray:
                """Simulate K steps for all N samples in parallel."""
                z_batch = cp.tile(z0, (N, 1))
                residuals = cp.empty((N, K), dtype=cp.float32)

                for k in range(K):
                    z_new = update_fn(z_batch, p_samples)
                    residuals[:, k] = cp.max(cp.abs(z_new - z_batch), axis=1)
                    z_batch = z_new

                return residuals

            # warm-up: triggers any lazy GPU compilation
            _simulate_batch(z0, p_samples)
            self._sync()

            # timed execution
            start = time.perf_counter()
            residuals = _simulate_batch(z0, p_samples)
            self._sync()
            elapsed = time.perf_counter() - start

        self._register_statistics(residuals, delta, epsilon, elapsed)
        return residuals
