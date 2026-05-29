import time
import typing
import warnings

try:
    import cupy as cp
    import cupyx.scipy.sparse as csp
    import cupyx.scipy.sparse.linalg as cpspla

    # Force CUDA initialization to happen NOW so we catch the error here
    cp.cuda.runtime.getDeviceCount()

    CUDA_AVAILABLE = True
except ImportError:
    import numpy as cp  # fallback to numpy
    import scipy.sparse as csp
    import scipy.sparse.linalg as cpspla

    CUDA_AVAILABLE = False


import numpy as np
import scipy.sparse as spa

from fomtools import LinearExpression
from fomtools.canonicalizers import Canonicalizer

if typing.TYPE_CHECKING:
    from fomtools import Step

from contextlib import nullcontext
from typing import Any


class Simulator(Canonicalizer):
    """
    An interface for simulating a Verification Problem using CuPy.

    Steps are recorded during canonicalization and compiled into efficient
    batched functions at solve time. This deferred-compilation approach
    allows an arbitrary sequence of steps to be registered incrementally
    (matching the MIP/SDP user experience) and then evaluated in a single
    forward pass over all parameter samples.

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

    step_descriptors : list[tuple[Step, Callable]]
        A list of (step, activation) tuples recorded during canonicalization.
        Each activation is a callable that applies the step-specific
        nonlinearity to the assembled rhs result.

    residual_specs : dict[int, tuple[list[LinearExpression], int | float | str]]
        A dictionary mapping step indices to (expr_list, norm_ord) tuples
        that define how residuals are computed at each step.

    device : int
        GPU device ID for CuPy execution.
    """

    def __init__(self, random_seed=0):
        super(Simulator, self).__init__(solver=None)
        self.param_to_samples = {}
        self.iterate_to_samples = {}

        self._seed_counter = random_seed
        self.step_descriptors = []
        self.residual_specs = {}
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
                    # Step output — register as iterate with zero init so
                    # subsequent steps can reference it via the value table.
                    self.iterate_to_samples[_var] = cp.zeros(_var.ndim, dtype=cp.float32)

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
                    slice indicating its location within the second dimension of q_samples."

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
                    slice indicating its location within the second dimension of q_samples.

        Returns
        -------
        cp.ndarray
            The value of the linear expression evaluated at the provided
            iterates and parameters.

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

    # ------------------------------------------------------------------
    # Canonicalize methods — record (step, activation) for deferred eval
    # ------------------------------------------------------------------

    def canonicalize_relu(self, step: "Step") -> None:
        """
        Record the provided ReLU step for deferred evaluation at solve time.

        The activation evaluates ReLU on the projected rhs coords selected by
        ``step.proj_indices`` and writes them into ``step.lhs_leaves[0]``.
        For full ReLU, ``proj_indices`` covers every rhs row and the fancy
        index is the identity slice. For partial-projection ReLU, non-proj
        rows of ``step.lhs`` share decomposition with ``step.rhs`` and are
        read off the existing rhs leaves at evaluation time — no separate
        storage needed.

        Parameters
        ----------
        step: Step
            A step object containing the ReLU operation.
        """
        proj_idx = np.array(sorted(step.proj_indices), dtype=int)

        def activation(result):
            # result shape: (N, rhs.ndim).
            # Output shape: (N, len(proj_idx)) — the value of lhs_leaves[0].
            return cp.maximum(result[:, proj_idx], 0)

        self.step_descriptors.append((step, activation))

    def canonicalize_softthreshold(self, step: "Step") -> None:
        """
        Record the provided soft-threshold step for deferred evaluation at solve time.

        Parameters
        ----------
        step: Step
            A step object containing the soft-threshold operation.
        """
        lambd = step.lambd

        def activation(result):
            return cp.sign(result) * cp.maximum(cp.abs(result) - lambd, 0)

        self.step_descriptors.append((step, activation))

    def canonicalize_saturatedlinear(self, step: "Step") -> None:
        """
        Record the provided saturated linear step for deferred evaluation at solve time.

        Parameters
        ----------
        step: Step
            A step object that represents a saturated linear operation.
        """
        l = step.l.lb if hasattr(step.l, "lb") else step.l
        u = step.u.ub if hasattr(step.u, "ub") else step.u

        def activation(result):
            return cp.clip(result, l, u)

        self.step_descriptors.append((step, activation))

    def canonicalize_affine(self, step: "Step") -> None:
        """
        Record the provided affine step for deferred evaluation at solve time.

        The LU factorization of ``step.lhs_mat`` is computed eagerly since
        it depends only on a constant matrix.

        Parameters
        ----------
        step: Step
            A step object that represents the affine transformation.
        """
        lhs_mat_cp = self._sparse_to_cp(step.lhs_mat)

        if csp.issparse(lhs_mat_cp):
            lu = cpspla.splu(lhs_mat_cp)

            def activation(rhs):
                return cp.array([lu.solve(r) for r in rhs])
        else:

            def activation(rhs):
                return cp.array([cp.linalg.solve(lhs_mat_cp, r) for r in rhs])

        self.step_descriptors.append((step, activation))

    _CANONICALIZERS = {
        "affine": canonicalize_affine,
        "relu": canonicalize_relu,
        "saturatedlinear": canonicalize_saturatedlinear,
        "softthreshold": canonicalize_softthreshold,
    }

    # ------------------------------------------------------------------
    # Objective methods — store residual specs keyed by step index
    # ------------------------------------------------------------------

    def set_zero_objective(self):
        """Record a zero residual at the most recently registered step."""
        self.residual_specs[len(self.step_descriptors) - 1] = ([], "zero")

    def set_infinity_norm_objective(self, expr_list, lb_list=None, ub_list=None):
        """Record an infinity-norm residual expression at the most recently registered step."""
        self.residual_specs[len(self.step_descriptors) - 1] = (expr_list, np.inf)

    def set_two_norm_objective(self, expr_list, lb_list=None, ub_list=None):
        """Record a squared two-norm residual expression at the most recently registered step."""
        self.residual_specs[len(self.step_descriptors) - 1] = (expr_list, 2)

    def set_one_norm_objective(self, expr_list):
        """Record a one-norm residual expression at the most recently registered step."""
        self.residual_specs[len(self.step_descriptors) - 1] = (expr_list, 1)

    # ------------------------------------------------------------------
    # Compilation — convert recorded steps/residuals to executable fns
    # ------------------------------------------------------------------

    def _compile_steps(self, simulation_data):
        """
        Compile all recorded step descriptors into executable functions.

        Called once at the start of ``solve_model`` after all variables are
        registered. Each step's rhs LinearExpression is converted to CuPy
        coefficients, and paired with the step's activation function.

        Parameters
        ----------
        simulation_data : dict
            Output of ``_prepare_simulation_data``.

        Returns
        -------
        list[tuple[Callable, slice, slice, Callable]]
            A list of ``(evaluate_fn, lhs_slice, prev_slice, compose_lhs)``
            tuples where ``evaluate_fn(z_stacked, p_stacked) -> cp.ndarray``
            computes the activation output to write into ``lhs_slice``, and
            ``compose_lhs(z_stacked, p_stacked) -> cp.ndarray`` evaluates
            the full composite ``step.lhs`` from the state vector (matters
            for partial-projection ReLU, where the activation only fills
            the fresh proj-leaf slot and the non-proj rows of ``step.lhs``
            are read off shared rhs leaves). ``prev_slice`` points to the
            previous iterate's slice in ``z_stacked`` for the default
            step-0 residual.
        """
        compiled = []
        z_slices = simulation_data["z_slices"]

        # Leaves a step *introduces* (where activation results get written).
        # For most steps this is the full ``lhs`` leaf; for partial-projection
        # Relu it is the smaller fresh proj-leaf returned by ``lhs_leaves``.
        step_leaves = [s.lhs_leaves[0] for s, _ in self.step_descriptors]

        for i, (step, activation) in enumerate(self.step_descriptors):
            coeffs = self._linear_expr_to_cupy(step.rhs)
            # For full ReLU/satlin/etc., ``step.lhs`` is a leaf and these
            # coeffs amount to ``identity @ z_stacked[:, lhs_slice]``, so
            # ``compose_lhs`` simply returns what was just written. For
            # partial-projection ReLU, the composite ``step.lhs`` mixes
            # the fresh proj-leaf with passthrough rows of ``step.rhs``;
            # evaluating this combination yields the full output.
            lhs_coeffs = self._linear_expr_to_cupy(step.lhs)
            lhs_slice = z_slices[step_leaves[i]]

            # Determine previous iterate slice for default residual computation.
            # Step 0: find the first original iterate leaf in the rhs decomposition.
            # Step k>0: previous step's composite output is provided by the caller
            # via the cached composed-lhs from the prior iteration.
            if i == 0:
                prev_slice = None
                for leaf in step.rhs.decomposition_dict:
                    if leaf in z_slices and leaf not in set(step_leaves):
                        prev_slice = z_slices[leaf]
                        break
                if prev_slice is None:
                    prev_slice = lhs_slice
            else:
                prev_slice = z_slices[step_leaves[i - 1]]

            def _make_fn(_coeffs=coeffs, _activation=activation):
                def evaluate_fn(z_stacked, p_stacked):
                    result = self._assemble_linear_combination(
                        z_stacked, p_stacked, _coeffs, simulation_data
                    )
                    return _activation(result)

                return evaluate_fn

            def _make_composer(_lhs_coeffs=lhs_coeffs):
                def compose_lhs(z_stacked, p_stacked):
                    return self._assemble_linear_combination(
                        z_stacked, p_stacked, _lhs_coeffs, simulation_data
                    )

                return compose_lhs

            compiled.append((_make_fn(), lhs_slice, prev_slice, _make_composer()))
        return compiled

    def _compile_residuals(self, simulation_data):
        """
        Compile all recorded residual specs into executable functions.

        Parameters
        ----------
        simulation_data : dict
            Output of ``_prepare_simulation_data``.

        Returns
        -------
        dict[int, Callable]
            A dictionary mapping step index to a callable
            ``residual_fn(z_stacked, p_stacked) -> cp.ndarray`` of shape ``(N,)``.
        """
        compiled = {}
        for step_idx, (expr_list, norm_ord) in self.residual_specs.items():
            if norm_ord == "zero":

                def _make_zero_fn():
                    def residual_fn(z_stacked, p_stacked):
                        return cp.zeros(z_stacked.shape[0], dtype=cp.float32)

                    return residual_fn

                compiled[step_idx] = _make_zero_fn()
            else:
                coeffs_list = [self._linear_expr_to_cupy(expr) for expr in expr_list]

                def _make_fn(_coeffs_list=coeffs_list, _norm_ord=norm_ord):
                    def residual_fn(z_stacked, p_stacked):
                        values = [
                            self._assemble_linear_combination(
                                z_stacked, p_stacked, c, simulation_data
                            )
                            for c in _coeffs_list
                        ]
                        concat = cp.concatenate(values, axis=1)

                        if _norm_ord == np.inf:
                            return cp.max(cp.abs(concat), axis=1)
                        elif _norm_ord == 2:
                            return cp.sum(concat**2, axis=1)
                        elif _norm_ord == 1:
                            return cp.sum(cp.abs(concat), axis=1)

                    return residual_fn

                compiled[step_idx] = _make_fn()
        return compiled

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve_model(
        self, delta: float = 0.05, epsilon: float = 0.0, device: str = None
    ) -> cp.ndarray:
        """
        Run a batched simulation of all registered steps.

        Each call re-simulates from scratch: the value table is rebuilt,
        all steps are compiled and executed in sequence, and residuals are
        computed. This method can be called repeatedly (e.g. inside the
        user's iteration loop) as new steps are registered.

        Parameters
        ----------
        delta: float, optional
            Confidence parameter for the risk bound.

        epsilon: float, optional
            Tolerance level defining the 0-1 error.

        device: str, optional
            Device to run the simulation on (``None``, ``"cpu"``, or ``"gpu"``).

        Returns
        -------
        residuals: cp.ndarray
           A 2D array of shape ``(num_samples, n_residuals)`` containing
           the residual at each step for each parameter sample.

        Raises
        ------
        ValueError
            If no steps have been registered.
        """
        if not self.step_descriptors:
            raise ValueError(
                "No steps have been registered with this simulator instance. "
                "Call vp.step.*() before solve()."
            )

        self._configure_device(device)
        simulation_data = self._prepare_simulation_data()
        compiled_steps = self._compile_steps(simulation_data)

        has_residual_specs = bool(self.residual_specs)
        if has_residual_specs:
            compiled_residuals = self._compile_residuals(simulation_data)
            n_residuals = len(compiled_residuals)
        else:
            compiled_residuals = None
            n_residuals = len(compiled_steps)

        with self._get_device_context(self.device):
            z0 = self._to_device(simulation_data["z0"])
            p_samples = self._to_device(simulation_data["p_samples"])
            N = p_samples.shape[0]

            def _simulate_batch(z0: cp.ndarray, p_samples: cp.ndarray) -> cp.ndarray:
                """Simulate all registered steps for all N samples in parallel."""
                z_batch = cp.tile(z0, (N, 1))
                residuals = cp.empty((N, n_residuals), dtype=cp.float32)
                residual_col = 0
                # Cache of the previous step's composite-lhs value, used by
                # the default residual path so that partial-projection ReLU
                # (where ``z_new`` only holds the projected outputs) still
                # produces a residual on the full composite output.
                prev_composed = None

                for step_idx, (
                    evaluate_fn,
                    lhs_slice,
                    prev_slice,
                    compose_lhs,
                ) in enumerate(compiled_steps):
                    z_new = evaluate_fn(z_batch, p_samples)
                    z_batch[:, lhs_slice] = z_new

                    if has_residual_specs:
                        if step_idx in compiled_residuals:
                            residuals[:, residual_col] = compiled_residuals[step_idx](
                                z_batch, p_samples
                            )
                            residual_col += 1
                    else:
                        # Default: inf-norm of composite_lhs[k] - composite_lhs[k-1].
                        # For full ReLU/satlin/etc., compose_lhs returns the freshly
                        # written leaf value (==z_new). For partial-projection ReLU,
                        # it assembles the full output (proj rows from the fresh leaf,
                        # non-proj rows from shared rhs leaves).
                        composed = compose_lhs(z_batch, p_samples)
                        if step_idx == 0:
                            prev_full = z_batch[:, prev_slice]
                        else:
                            prev_full = prev_composed
                        residuals[:, residual_col] = cp.max(
                            cp.abs(composed - prev_full), axis=1
                        )
                        prev_composed = composed
                        residual_col += 1

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
