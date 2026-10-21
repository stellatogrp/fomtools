import time
import typing
import warnings

try:
    import cupy as cp
    import cupyx.scipy.sparse as csp
    import cupyx.scipy.sparse.linalg as cpspla

    # Force CUDA initialization NOW so a missing/broken GPU falls back to numpy here
    # rather than surfacing deep inside a solve. getDeviceCount raises a cupy
    # CUDARuntimeError (NOT an ImportError) when no usable device exists; the previous
    # ``except ImportError`` silently let that propagate. The two failure layers are
    # split: import errors (cupy absent / broken install) below, then a broad catch
    # for any CUDA-init failure (intentionally broad — any failure means "use numpy").
    try:
        cp.cuda.runtime.getDeviceCount()
        CUDA_AVAILABLE = True
    except Exception:  # noqa: BLE001 - any CUDA init failure -> numpy fallback
        raise ImportError("CUDA device unavailable") from None
except (ImportError, ModuleNotFoundError, AttributeError):
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


class _ActivationStep:
    """
    Minimal step holder for subproblem activations that have no ``Step`` class.

    Projection / QP / LP proxes are encoded via KKT on the MIP/SDP backends, so
    there is no concrete ``Step`` to record. This lightweight object exposes the
    three attributes the simulator's ``_compile_steps`` reads off a step — ``arg``
    (the assembled input expression), ``output`` (the fresh leaf the activation
    writes), and ``output_leaves`` — so such proxes flow through the same
    ``step_descriptors`` machinery as ``Relu``/``Affine``.
    """

    def __init__(self, arg: LinearExpression, output: LinearExpression):
        self.arg = arg
        self.output = output
        self.output_leaves = [output]


# ----------------------------------------------------------------------------
# Subproblem evaluators (activation factories for the step decomposer)
#
# Implicit-iterate proxes/steps whose output is defined by a KKT system with
# *complementarity* (the active set varies per sample) cannot be evaluated with a
# single stacked linear solve. These build the ``activation(batch) -> out``
# closures consumed by ``Simulator.register_subproblem_step``: batched solver-free
# projections (simplex / ℓ1-ball / ℓ∞-norm), Condat's O(n) total-variation
# algorithm, and per-sample parameterized cvxpy solves (polyhedral projection,
# inequality QP/LP). ``batch`` is the assembled input ``(N, d)`` and ``out`` is
# ``(N, n)``.
# ----------------------------------------------------------------------------


def _xp(arr):
    """Return the array module (cupy or numpy) backing ``arr``."""
    if CUDA_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp
    return np


def _to_numpy(arr) -> np.ndarray:
    """
    Move ``arr`` to host numpy (no-op if already numpy).

    Used by the CPU-side activations (``tv_activation``, ``polyhedral_activation``,
    ``qp_activation``) whose inner loops run on host (Condat / cvxpy-CLARABEL). On a
    GPU run this pairs with `_to_like` to bracket each activation with a
    device→host→device transfer. Since these activations run inside
    ``_simulate_batch`` — invoked once per ``solve_model`` (the warm-up is skipped off
    GPU) — a mixed CUDA + cvxpy pipeline incurs that PCIe round-trip every iteration
    and can become bus-bound. This is inherent: cvxpy/CLARABEL have no GPU backend.
    """
    if CUDA_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def _to_like(host: np.ndarray, ref):
    """Move host numpy ``host`` back to the device of ``ref`` (see `_to_numpy`
    for the GPU round-trip cost this incurs in the CPU-side activations)."""
    if CUDA_AVAILABLE and isinstance(ref, cp.ndarray):
        return cp.asarray(host)
    return host


def _warn_if_infeasible(out: np.ndarray, label: str) -> None:
    """Warn (once) if any per-sample cvxpy subproblem was infeasible/unbounded.

    Such samples are written as ``NaN`` (cvxpy returns ``None``). Left unflagged the
    NaN propagates through later steps and surfaces as a deceptively "zero" residual
    after the ``max(abs(...))`` comparison, so emit a count to aid diagnosis.
    """
    n_bad = int(np.isnan(out).any(axis=1).sum())
    if n_bad:
        warnings.warn(
            f"{label}: {n_bad} of {out.shape[0]} samples were infeasible/unbounded "
            "(cvxpy returned no solution); those rows are NaN.",
            RuntimeWarning,
            stacklevel=2,
        )


def project_simplex_batch(V, a: float):
    r"""
    Euclidean projection of each row of ``V`` onto ``{u >= 0, 1^T u = a}``.

    Vectorized Held–Wolfe / Duchi (2008) sort algorithm: for ``a > 0`` the
    threshold ``theta`` is found per row from the descending sort and cumulative
    sums, and the projection is ``max(V - theta, 0)``. No Python loop over the
    batch.

    Parameters
    ----------
    V : (N, n) array (numpy or cupy)
        Points to project (one per row).
    a : float
        Positive simplex sum.

    Returns
    -------
    (N, n) array
        The row-wise projections.
    """
    xp = _xp(V)
    N, n = V.shape
    U = xp.flip(xp.sort(V, axis=1), axis=1)  # each row sorted descending
    cssv = xp.cumsum(U, axis=1) - a
    ind = xp.arange(1, n + 1)
    rho = xp.sum((U - cssv / ind) > 0, axis=1)  # >= 1 since a > 0
    theta = cssv[xp.arange(N), rho - 1] / rho
    return xp.maximum(V - theta[:, None], 0.0)


def project_l1_ball_batch(V, r: float):
    r"""
    Euclidean projection of each row of ``V`` onto ``{u : ||u||_1 <= r}``.

    Rows already inside the ball are returned unchanged; rows outside are
    projected via ``sign(V) * project_simplex_batch(|V|, r)``. No Python loop.

    Parameters
    ----------
    V : (N, n) array (numpy or cupy)
        Points to project.
    r : float
        Ball radius (``>= 0``).

    Returns
    -------
    (N, n) array
        The row-wise projections.
    """
    xp = _xp(V)
    if r <= 0.0:
        return xp.zeros_like(V)
    absV = xp.abs(V)
    over = absV.sum(axis=1) > r
    w = xp.sign(V) * project_simplex_batch(absV, r)
    return xp.where(over[:, None], w, V)


def simplex_activation(a: float):
    """Activation projecting each sample onto the scaled simplex."""
    return lambda V: project_simplex_batch(V, a)


def l1_ball_activation(r: float):
    """Activation projecting each sample onto the ℓ1-ball of radius ``r``."""
    return lambda V: project_l1_ball_batch(V, r)


def norm_inf_activation(radius: float):
    """Activation for the ℓ∞-norm prox via Moreau: ``v - proj_{||.||_1 <= radius}(v)``."""
    return lambda V: V - project_l1_ball_batch(V, radius)


def condat_tv_1d(y: np.ndarray, lam: float) -> np.ndarray:
    r"""
    1-D total-variation prox via Condat's O(n) direct algorithm.

    Computes ``argmin_x 1/2 ||x - y||^2 + lam * sum_i |x_{i+1} - x_i|`` exactly in a
    single forward pass (Condat, "A Direct Algorithm for 1-D Total Variation
    Denoising", 2013). ``lam <= 0`` or length ``< 2`` returns a copy of ``y``.

    Parameters
    ----------
    y : (n,) np.ndarray
        Input signal.
    lam : float
        Total-variation weight.

    Returns
    -------
    (n,) np.ndarray
        The proximal point.
    """
    y = np.asarray(y, dtype=float)
    n = y.shape[0]
    if n < 2 or lam <= 0.0:
        return y.copy()

    x = np.empty(n, dtype=float)
    k = 0  # start index of the current segment under construction
    k0 = 0  # current sample being processed
    umin = lam
    umax = -lam
    vmin = y[0] - lam
    vmax = y[0] + lam
    kplus = 0  # last index where the upper slack saturated
    kminus = 0  # last index where the lower slack saturated

    while True:
        if k0 == n - 1:
            # Reached the end of the signal: flush the remaining segment.
            #
            # Each flush branch below resets the tracker/envelope of its OWN side but
            # leaves the opposite one (kplus/vmax after a umin flush, kminus/vmin after
            # a umax flush) stale. This is safe: the flush sets its own slack to +/-lam,
            # so on re-entry the same-side test fails and — by the algorithm's envelope
            # invariant — the opposite flush is never taken with a stale tracker (an
            # empty x[k:kplus+1] write with kplus < k cannot occur). Verified by
            # exhaustive search over short integer signals; matches Condat's reference,
            # which likewise does not reset the opposite side here.
            if umin < 0.0:
                x[k : kminus + 1] = vmin
                k = kminus = kminus + 1
                if k >= n:
                    # The flush filled x[: n]; the new segment would start past the
                    # end, so guard the y[k] read and finish.
                    return x
                k0 = k
                vmin = y[k]
                umin = lam
                umax = y[k] + lam - vmax
            elif umax > 0.0:
                x[k : kplus + 1] = vmax
                k = kplus = kplus + 1
                if k >= n:
                    return x
                k0 = k
                vmax = y[k]
                umax = -lam
                umin = y[k] - lam - vmin
            else:
                x[k:n] = vmin + umin / (k0 - k + 1)
                return x
        else:
            if y[k0 + 1] + umin < vmin - lam:
                # negative jump
                x[k : kminus + 1] = vmin
                k = k0 = kplus = kminus = kminus + 1
                vmin = y[k]
                vmax = y[k] + 2 * lam
                umin = lam
                umax = -lam
            elif y[k0 + 1] + umax > vmax + lam:
                # positive jump
                x[k : kplus + 1] = vmax
                k = k0 = kplus = kminus = kplus + 1
                vmax = y[k]
                vmin = y[k] - 2 * lam
                umin = lam
                umax = -lam
            else:
                # no jump yet: advance and update the slacks
                k0 += 1
                umin += y[k0] - vmin
                umax += y[k0] - vmax
                if umin >= lam:
                    vmin += (umin - lam) / (k0 - k + 1)
                    umin = lam
                    kminus = k0
                if umax <= -lam:
                    vmax += (umax + lam) / (k0 - k + 1)
                    umax = -lam
                    kplus = k0


def tv_activation(lam: float):
    """Activation applying Condat's TV prox to each sample (looped on host).

    Host-side loop: on a GPU run the batch is moved device→host and the result back
    (see `_to_numpy`).
    """

    def activation(V):
        Vnp = _to_numpy(V)
        out = np.stack([condat_tv_1d(row, lam) for row in Vnp])
        return _to_like(out, V)

    return activation


def polyhedral_activation(A: np.ndarray, b, n: int, m: int, b_is_param: bool):
    r"""
    Activation projecting each sample onto ``{u : A u <= b}`` with cvxpy.

    Builds a single parameterized (DPP) projection problem and solves it per sample.
    The input batch has the point in its first ``n`` columns; when ``b`` is a
    parameter (``b_is_param``), the next ``m`` columns carry the per-sample ``b``.

    Parameters
    ----------
    A : (m, n) np.ndarray
        Inequality matrix.
    b : (m,) np.ndarray or None
        Fixed right-hand side when ``b`` is not a parameter.
    n, m : int
        Point and constraint dimensions.
    b_is_param : bool
        Whether ``b`` varies per sample (stacked into the input batch).

    Notes
    -----
    The per-sample loop runs on host (cvxpy/CLARABEL has no GPU backend); on a GPU run
    each call moves the batch device→host and the result back (see `_to_numpy`).
    """
    import cvxpy as cp_solver

    u = cp_solver.Variable(n)
    x_par = cp_solver.Parameter(n)
    if b_is_param:
        b_par = cp_solver.Parameter(m)
        constraints = [A @ u <= b_par]
    else:
        b_par = None
        constraints = [A @ u <= np.asarray(b, dtype=float)]
    prob = cp_solver.Problem(cp_solver.Minimize(cp_solver.sum_squares(u - x_par)), constraints)

    def activation(V):
        Vnp = _to_numpy(V)
        out = np.empty((Vnp.shape[0], n), dtype=float)
        for i in range(Vnp.shape[0]):
            x_par.value = Vnp[i, :n]
            if b_is_param:
                b_par.value = Vnp[i, n : n + m]
            prob.solve(solver=cp_solver.CLARABEL)
            out[i] = u.value if u.value is not None else np.nan
        _warn_if_infeasible(out, "IndPolyhedral projection (simulator)")
        return _to_like(out, V)

    return activation


def qp_activation(P, A, C, n: int, p: int, m: int, is_lp: bool):
    r"""
    Activation solving a QP/LP per sample with cvxpy.

    Builds a single parameterized (DPP) problem ``minimize 0.5 x^T P x + q^T x s.t.
    A x = b, C x <= d`` (the quadratic term is dropped when ``is_lp``) and solves it
    per sample. The input batch stacks the present parametric blocks in the order
    ``[q (n) | b (p) | d (m)]``.

    Parameters
    ----------
    P : (n, n) array or None
        Quadratic term (ignored when ``is_lp``).
    A : (p, n) array or None
        Equality matrix (``None`` if absent).
    C : (m, n) array or None
        Inequality matrix (``None`` if absent).
    n, p, m : int
        Primal / equality / inequality dimensions.
    is_lp : bool
        Drop the quadratic term (linear program).

    Notes
    -----
    The per-sample loop runs on host (cvxpy/CLARABEL has no GPU backend); on a GPU run
    each call moves the batch device→host and the result back (see `_to_numpy`).
    """
    import cvxpy as cp_solver

    x = cp_solver.Variable(n)
    q_par = cp_solver.Parameter(n)
    objective = q_par @ x
    if not is_lp:
        P_dense = P.toarray() if spa.issparse(P) else np.asarray(P, dtype=float)
        objective = 0.5 * cp_solver.quad_form(x, cp_solver.psd_wrap(P_dense)) + objective

    constraints = []
    b_par = d_par = None
    if A is not None:
        b_par = cp_solver.Parameter(p)
        A_mat = A.toarray() if spa.issparse(A) else np.asarray(A, dtype=float)
        constraints.append(A_mat @ x == b_par)
    if C is not None:
        d_par = cp_solver.Parameter(m)
        C_mat = C.toarray() if spa.issparse(C) else np.asarray(C, dtype=float)
        constraints.append(C_mat @ x <= d_par)
    prob = cp_solver.Problem(cp_solver.Minimize(objective), constraints)

    def activation(V):
        Vnp = _to_numpy(V)
        out = np.empty((Vnp.shape[0], n), dtype=float)
        for i in range(Vnp.shape[0]):
            row = Vnp[i]
            q_par.value = row[:n]
            off = n
            if A is not None:
                b_par.value = row[off : off + p]
                off += p
            if C is not None:
                d_par.value = row[off : off + m]
            prob.solve(solver=cp_solver.CLARABEL)
            out[i] = x.value if x.value is not None else np.nan
        _warn_if_infeasible(out, "QP/LP solve (simulator)")
        return _to_like(out, V)

    return activation


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
        nonlinearity to the assembled arg result.

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
        self.user_iterates = {}
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
        Convert a NumPy or SciPy sparse matrix to a CuPy array
        or cupyx CSR sparse matrix, respectively.

        Parameters
        ----------
        matrix: np.ndarray | scipy.sparse.spmatrix
            The sparse matrix to be converted.

        Returns
        -------
        cp.ndarray | cupyx.scipy.sparse.csr_matrix
            A CuPy array or cupyx CSR sparse matrix containing the same data as
            the input matrix.
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
            terms) to lists of tuples (leaf, coeff_cp), where leaf identifies
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
                    f"Both bounds and samples are provided for the {var_type}; bounds will be ignored."
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
                    samples = np.asarray(samples)
                    self.param_to_samples[_var] = cp.array(samples, dtype=cp.float32)
                    if not has_bounds:
                        _var.lb = np.min(samples, axis=0)
                        _var.ub = np.max(samples, axis=0)
                elif has_bounds and np.array_equal(lb, ub):
                    # A user parameter pinned by equal bounds represents a
                    # constant value. Store that value so symbolic expressions
                    # involving the parameter evaluate correctly per sample.
                    self.iterate_to_samples[_var] = cp.array(lb, dtype=cp.float32)
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
        Parameter samples are concatenated along axis 1 into ``p_samples``.
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
                    slice indicating its location within the second dimension of ``p_samples``.

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
        r"""
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
            Concatenation of all iterate variables. Individual iterate
            values are retrieved using slices stored in ``simulation_data["z_slices"]``.

        p_stacked: cp.ndarray
            Concatenation of all parameter variables. Individual parameter values
            are retrieved using slices stored in ``simulation_data["p_slices"]``.

        coeffs: dict[str, list[tuple[LinearExpression, cp.ndarray | csp.csr_matrix]]]
            A dictionary mapping term categories (e.g., iterate or parameter
            terms) to lists of tuples (leaf, coeff_cp), where leaf identifies
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
                    slice indicating its location within the second dimension of ``p_samples``.

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

        The activation evaluates ReLU on the projected arg coords selected by
        ``step.proj_indices`` and writes them into ``step.output_leaves[0]``.
        For full ReLU, ``proj_indices`` covers every arg row and the fancy
        index is the identity slice. For partial-projection ReLU, non-proj
        rows of ``step.output`` share decomposition with ``step.arg`` and are
        read off the existing arg leaves at evaluation time — no separate
        storage needed.

        Parameters
        ----------
        step: Step
            A step object containing the ReLU operation.
        """
        proj_idx = np.array(sorted(step.proj_indices), dtype=int)

        def activation(result):
            # result shape: (N, arg.ndim).
            # Output shape: (N, len(proj_idx)) — the value of output_leaves[0].
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

        The LU factorization of ``step.output_mat`` is computed eagerly since
        it depends only on a constant matrix.

        Parameters
        ----------
        step: Step
            A step object that represents the affine transformation.
        """
        output_mat_cp = self._sparse_to_cp(step.output_mat)

        if csp.issparse(output_mat_cp):
            # Factor once (the matrix is constant) and solve all N right-hand
            # sides in a single multi-RHS triangular solve. ``arg`` is (N, d);
            # SuperLU.solve accepts a 2-D RHS of shape (d, N) -> (d, N).
            lu = cpspla.splu(output_mat_cp.tocsc())

            def activation(arg):
                return lu.solve(arg.T).T
        else:

            def activation(arg):
                # cp.linalg.solve broadcasts over the columns of the (d, N) RHS,
                # so all samples are solved at once (no per-sample Python loop).
                return cp.linalg.solve(output_mat_cp, arg.T).T

        self.step_descriptors.append((step, activation))

    _CANONICALIZERS = {
        "affine": canonicalize_affine,
        "relu": canonicalize_relu,
        "saturatedlinear": canonicalize_saturatedlinear,
        "softthreshold": canonicalize_softthreshold,
    }

    def register_subproblem_step(
        self, arg: LinearExpression, output_leaf: LinearExpression, activation
    ) -> None:
        """
        Record a subproblem step for deferred evaluation at solve time.

        Used by the step decomposer for proxes/steps that have no ``Step`` class
        on the simulator (projections onto the simplex / ℓ1-ball, the ℓ∞-norm and
        total-variation proxes, polyhedral projections, and inequality QP/LP). The
        ``activation`` is a callable mapping the assembled input batch
        ``(N, arg.ndim)`` to the output batch ``(N, output_leaf.ndim)``; the result
        is written into ``output_leaf``'s slot. See the subproblem activation
        factories defined in this module (e.g. `simplex_activation`,
        `qp_activation`).

        Parameters
        ----------
        arg : LinearExpression
            The assembled per-sample input to the subproblem.
        output_leaf : LinearExpression
            The fresh leaf the activation result is written into.
        activation : Callable
            Maps the input batch to the output batch.
        """
        self.step_descriptors.append((_ActivationStep(arg, output_leaf), activation))

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
        registered. Each step's arg LinearExpression is converted to CuPy
        coefficients, and paired with the step's activation function.

        Parameters
        ----------
        simulation_data : dict
            Output of ``_prepare_simulation_data``.

        Returns
        -------
        list[tuple[Callable, slice, slice, Callable]]
            A list of ``(evaluate_fn, output_slice, prev_slice, compose_output)``
            tuples where ``evaluate_fn(z_stacked, p_stacked) -> cp.ndarray``
            computes the activation output to write into ``output_slice``, and
            ``compose_output(z_stacked, p_stacked) -> cp.ndarray`` evaluates
            the full composite ``step.output`` from the state vector (matters
            for partial-projection ReLU, where the activation only fills
            the fresh proj-leaf slot and the non-proj rows of ``step.output``
            are read off shared arg leaves). ``prev_slice`` points to the
            previous iterate's slice in ``z_stacked`` for the default
            step-0 residual.
        """
        compiled = []
        z_slices = simulation_data["z_slices"]

        # Leaves a step *introduces* (where activation results get written).
        # For most steps this is the full ``output`` leaf; for partial-projection
        # Relu it is the smaller fresh proj-leaf returned by ``output_leaves``.
        # The simulator writes one ``z_new`` slot per step, so we require
        # ``output_leaves`` to be a single-leaf list. If a future step type
        # introduces multiple fresh leaves per call, _compile_steps must be
        # extended to allocate and write a slot per leaf — fail loudly here
        # rather than silently dropping the extra leaves.
        step_leaves = []
        for s, _ in self.step_descriptors:
            leaves = s.output_leaves
            if len(leaves) != 1:
                raise NotImplementedError(
                    f"{type(s).__name__}.output_leaves returned {len(leaves)} leaves; "
                    "Simulator._compile_steps assumes exactly one fresh leaf per step. "
                    "Extend the compile/write loop to handle multi-leaf steps."
                )
            step_leaves.append(leaves[0])

        for i, (step, activation) in enumerate(self.step_descriptors):
            coeffs = self._linear_expr_to_cupy(step.arg)
            # For full ReLU/satlin/etc., ``step.output`` is a leaf and these
            # coeffs amount to ``identity @ z_stacked[:, output_slice]``, so
            # ``compose_output`` simply returns what was just written. For
            # partial-projection ReLU, the composite ``step.output`` mixes
            # the fresh proj-leaf with passthrough rows of ``step.arg``;
            # evaluating this combination yields the full output.
            output_coeffs = self._linear_expr_to_cupy(step.output)
            output_slice = z_slices[step_leaves[i]]

            # Determine previous iterate slice for default residual computation.
            # Step 0: find the first original iterate leaf in the arg decomposition.
            # Step k>0: previous step's composite output is provided by the caller
            # via the cached composed-output from the prior iteration.
            if i == 0:
                prev_slice = None
                for leaf in step.arg.decomposition_dict:
                    if leaf in z_slices and leaf not in set(step_leaves):
                        prev_slice = z_slices[leaf]
                        break
                if prev_slice is None:
                    prev_slice = output_slice
            else:
                prev_slice = z_slices[step_leaves[i - 1]]

            def _make_fn(_coeffs=coeffs, _activation=activation):
                def evaluate_fn(z_stacked, p_stacked):
                    result = self._assemble_linear_combination(
                        z_stacked, p_stacked, _coeffs, simulation_data
                    )
                    return _activation(result)

                return evaluate_fn

            def _make_composer(_output_coeffs=output_coeffs):
                def compose_output(z_stacked, p_stacked):
                    return self._assemble_linear_combination(
                        z_stacked, p_stacked, _output_coeffs, simulation_data
                    )

                return compose_output

            compiled.append((_make_fn(), output_slice, prev_slice, _make_composer()))
        return compiled

    def record_iterate(self, expression):
        """Record a public operation's output after all its internal steps."""
        self.user_iterates[len(self.step_descriptors) - 1] = expression

    def _compile_residuals(self, simulation_data, specs=None):
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
        for step_idx, (expr_list, norm_ord) in (
            self.residual_specs if specs is None else specs
        ).items():
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
            Tolerance level defining the 0–1 error.

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

        specs = self.residual_specs
        if not specs:
            specs = {}
            previous = None
            for idx, expr in self.user_iterates.items():
                if previous is not None and idx >= 0:
                    if expr.ndim != previous.ndim:
                        raise ValueError(
                            "Default residuals require equal-sized user iterates; set an explicit objective."
                        )
                    specs[idx] = ([expr - previous], np.inf)
                previous = expr
            if not specs:
                raise ValueError(
                    "Default residuals require an initial iterate; call Verifier.iterate or set an explicit objective."
                )
        compiled_residuals = self._compile_residuals(simulation_data, specs)
        n_residuals = len(compiled_residuals)

        with self._get_device_context(self.device):
            z0 = self._to_device(simulation_data["z0"])
            p_samples = self._to_device(simulation_data["p_samples"])
            N = p_samples.shape[0]

            def _simulate_batch(z0: cp.ndarray, p_samples: cp.ndarray) -> cp.ndarray:
                """Simulate all registered steps for all N samples in parallel."""
                z_batch = cp.tile(z0, (N, 1))
                residuals = cp.empty((N, n_residuals), dtype=cp.float32)
                residual_col = 0

                for step_idx, (
                    evaluate_fn,
                    output_slice,
                    prev_slice,
                    compose_output,
                ) in enumerate(compiled_steps):
                    z_new = evaluate_fn(z_batch, p_samples)
                    z_batch[:, output_slice] = z_new

                    if step_idx in compiled_residuals:
                        residuals[:, residual_col] = compiled_residuals[step_idx](
                            z_batch, p_samples
                        )
                        residual_col += 1

                return residuals

            # Warm-up: trigger CuPy's lazy GPU kernel compilation so it is excluded
            # from the timed run. Only relevant on GPU — CLARABEL and the CPU-side
            # activations (TV/polyhedral/QP/LP loops) have no lazy-compile phase, so
            # warming up there would just double the solve time for no benefit.
            if CUDA_AVAILABLE:
                _simulate_batch(z0, p_samples)
                self._sync()

            # timed execution
            start = time.perf_counter()
            residuals = _simulate_batch(z0, p_samples)
            self._sync()
            elapsed = time.perf_counter() - start

        self._register_statistics(residuals, delta, epsilon, elapsed)
        return residuals
