from functools import cached_property

import numpy as np
import scipy.sparse as spa

from fomtools.linearexpression import LinearExpression
from fomtools.steps.step import Step

# Tolerance for the symmetry and PSD checks on ``P``. PSD allows a small negative
# eigenvalue to absorb floating-point noise (e.g. ``A.T @ A`` Hessians).
_PSD_TOL = 1e-8


def _is_linexpr(value) -> bool:
    """EAFP-free test for a ``LinearExpression`` (codebase convention: no isinstance)."""
    return hasattr(value, "decomposition_dict")


def _vec_dim(value, fallback: int) -> int:
    """Return the vector dimension of a scalar/array/``LinearExpression`` parameter."""
    if value is None:
        return fallback
    if _is_linexpr(value):
        return value.ndim
    arr = np.asarray(value, dtype=float)
    return fallback if arr.ndim == 0 else arr.shape[0]


# Below this dimension, sparse matrices are densified for the eigenvalue check
# (``eigsh`` is fragile / cannot return all-but-one eigenvalues on small matrices).
_DENSE_EIG_MAX_N = 200


def _min_eig(P) -> float:
    """Smallest eigenvalue of a symmetric matrix ``P`` (dense or sparse)."""
    if spa.issparse(P):
        if P.nnz == 0:  # all-zero (e.g. LinearProgram); trivially PSD, skip eigsh
            return 0.0
        n = P.shape[0]
        if n <= _DENSE_EIG_MAX_N:
            return float(np.linalg.eigvalsh(P.toarray()).min())
        try:
            return float(
                spa.linalg.eigsh(P.asfptype(), k=1, which="SA", return_eigenvectors=False)[0]
            )
        except Exception:  # pragma: no cover - convergence fallback
            return float(np.linalg.eigvalsh(P.toarray()).min())
    return float(np.linalg.eigvalsh(np.asarray(P, dtype=float)).min())


def _sym_error(P) -> float:
    """Max absolute asymmetry ``max|P - P^T|`` (dense or sparse)."""
    diff = P - P.T
    if spa.issparse(diff):
        diff = diff.tocoo()
        return 0.0 if diff.nnz == 0 else float(np.abs(diff.data).max())
    return float(np.abs(np.asarray(diff, dtype=float)).max())


class QuadraticProgram(Step):
    r"""
    Step representing the solution of an arbitrary convex quadratic program

    $$
        x^\star = \arg\min_x\; \tfrac12 x^\top P x + q^\top x
        \quad\text{s.t.}\quad A x = b,\; C x \le d,
    $$

    where :math:`P \in \mathbf{R}^{n\times n}` is a fixed **symmetric PSD** matrix
    (dense ``numpy`` or ``scipy.sparse``), and ``q``, ``b``, ``d`` may be numeric
    data or :class:`LinearExpression` parameters (carrying dependence on prior
    iterates). The verification-side lowering (``decompose_quadraticprogram`` in
    the step decomposer) posts the KKT system

    $$
        P x + q + A^\top \nu + C^\top \lambda = 0,\quad
        A x = b,\quad C x \le d,\quad
        0 \le \lambda \perp (d - C x) \ge 0,
    $$

    with :math:`\nu` a free equality multiplier and :math:`\lambda \ge 0`, using
    the complementarity primitive. The MIP big-M encoding of the complementarity
    needs a finite upper bound on :math:`\lambda`; supply it via ``dual_ub``
    whenever inequalities are present (OBBT tightens it further when enabled).

    Invoked via :meth:`Verifier.quadratic_program` rather than the single-``arg``
    ``run_step`` path: an arbitrary QP has up to three parametric inputs
    (``q``, ``b``, ``d``), and each ``LinearExpression`` already references the
    prior iterates it was built from, so no input is forced into an ``arg`` slot.

    Parameters
    ----------
    P : np.ndarray or scipy.sparse matrix
        The ``(n, n)`` quadratic term; must be symmetric and PSD.
    q : None, float, np.ndarray, or LinearExpression, optional
        Linear cost of dimension ``n``. If ``None`` (projection convenience),
        ``q`` defaults to ``-arg`` in the decomposer, which requires ``arg`` to be
        a dim-``n`` ``LinearExpression``.
    A : np.ndarray or scipy.sparse, optional
        Equality matrix of shape ``(p, n)``. ``None`` for no equality constraints.
    b : float, np.ndarray, or LinearExpression, optional
        Equality right-hand side of dimension ``p``.
    C : np.ndarray or scipy.sparse, optional
        Inequality matrix of shape ``(m, n)``. ``None`` for no inequalities.
    d : float, np.ndarray, or LinearExpression, optional
        Inequality right-hand side of dimension ``m``.
    dual_ub : None, float, or np.ndarray, optional
        Finite upper bound(s) on the inequality dual ``lambda >= 0``. Required
        (raised eagerly in the decomposer) whenever ``C`` is present.
    primal_bounds : tuple[np.ndarray, np.ndarray], optional
        ``(lb, ub)`` on the primal ``x``. Needed when ``P`` is singular / not
        positive definite (the stationarity row cannot pin ``x`` bounds);
        otherwise derived analytically and tightened by OBBT.
    nu_bounds : tuple[np.ndarray, np.ndarray], optional
        ``(lb, ub)`` on the free equality multiplier ``nu``. Defaults to
        ``+/- inf`` (``nu`` only appears in the linear stationarity row).
    arg : LinearExpression, optional
        The projection point used when ``q is None`` (then ``q := -arg``).
        Optional otherwise; retained only as the nominal Step input.
    relax_binary_vars : bool, default False
        Forwarded to the Step interface (relax big-M binaries when set).
    """

    skip_obbt: bool = True

    def __init__(
        self,
        P,
        q=None,
        A=None,
        b=None,
        C=None,
        d=None,
        dual_ub=None,
        primal_bounds=None,
        nu_bounds=None,
        arg: LinearExpression = None,
        relax_binary_vars: bool = False,
    ):
        # --- validate P: square, symmetric, PSD (dense or sparse) ---
        if P.ndim != 2 or P.shape[0] != P.shape[1]:
            raise ValueError(f"P must be a square 2-D matrix; got shape {P.shape}")
        if spa.issparse(P):
            P = P.tocsr()  # CSR supports T, arithmetic, and @ (LinearExpression.__rmatmul__)
        n = P.shape[0]
        sym_err = _sym_error(P)
        if sym_err > _PSD_TOL:
            raise ValueError(f"P must be symmetric; max|P - P^T| = {sym_err:.3e}")
        min_eig = _min_eig(P)
        if min_eig < -_PSD_TOL:
            raise ValueError(
                f"P must be positive semidefinite; smallest eigenvalue {min_eig:.3e} < -{_PSD_TOL:g}"
            )

        # --- validate parametric data dimensions ---
        if q is None:
            if arg is None or not _is_linexpr(arg) or arg.ndim != n:
                raise ValueError(
                    "q=None defaults to -arg, which requires arg to be a "
                    f"LinearExpression of dimension n={n}."
                )
        elif _vec_dim(q, n) != n:
            raise ValueError(f"q dim {_vec_dim(q, n)} must match P dimension n={n}")

        if A is not None:
            if A.shape[1] != n:
                raise ValueError(f"A must have {n} columns to match P; got shape {A.shape}")
            p = A.shape[0]
            if b is None:
                raise ValueError("b must be provided when A is given")
            if _vec_dim(b, p) != p:
                raise ValueError(f"b dim {_vec_dim(b, p)} must match A rows {p}")
        if C is not None:
            if C.shape[1] != n:
                raise ValueError(f"C must have {n} columns to match P; got shape {C.shape}")
            m = C.shape[0]
            if d is None:
                raise ValueError("d must be provided when C is given")
            if _vec_dim(d, m) != m:
                raise ValueError(f"d dim {_vec_dim(d, m)} must match C rows {m}")

        # Bypass Step.__init__ (it dereferences arg.ndim): arg is optional here.
        self.arg = arg
        self.relax_binary_vars = relax_binary_vars
        self.n = n
        self.proj_indices = set(range(n))

        self.P = P
        self.q = q
        self.A = A
        self.b = b
        self.C = C
        self.d = d
        self.dual_ub = dual_ub
        self.primal_bounds = primal_bounds
        self.nu_bounds = nu_bounds

    @cached_property
    def output(self) -> LinearExpression:
        """Output is the QP primal solution ``x`` of dimension ``n`` (``!= arg.ndim``)."""
        return LinearExpression.from_int(self.n)

    def update_bounds(self, lb, ub):
        """Store primal bounds (interface compliance; QP is a composite step that
        bypasses ``_process_primitive``, so this is rarely invoked)."""
        self.lb, self.ub = lb, ub
        self.output.update_bounds(lb=lb, ub=ub)

    def __hash__(self):
        return hash((self.arg, self.output))

    def solve(self, q=None, b=None, d=None) -> np.ndarray:
        r"""
        Numerically solve the QP with cvxpy (reference forward-evaluation).

        Numeric overrides for ``q``/``b``/``d`` may be supplied for the case where
        the stored values are ``LinearExpression`` parameters; otherwise the stored
        numeric data is used. ``P`` is densified for cvxpy.

        Returns
        -------
        np.ndarray
            The optimal primal ``x`` of dimension ``n``.
        """
        import cvxpy as cp

        def _resolve(override, stored, name):
            value = override if override is not None else stored
            if value is None:
                return None
            if _is_linexpr(value):
                raise ValueError(
                    f"{name} is a LinearExpression; pass a numeric override to solve()"
                )
            return np.asarray(value, dtype=float)

        P_dense = P.toarray() if spa.issparse(P := self.P) else np.asarray(P, dtype=float)
        q_val = _resolve(q, self.q, "q")
        if q_val is None:
            raise ValueError("q is None (defaults to -arg); pass a numeric q to solve()")
        q_val = np.broadcast_to(q_val, (self.n,)).astype(float)

        x = cp.Variable(self.n)
        objective = q_val @ x
        if np.any(P_dense):
            objective = 0.5 * cp.quad_form(x, cp.psd_wrap(P_dense)) + objective
        constraints = []
        if self.A is not None:
            A = self.A.toarray() if spa.issparse(self.A) else np.asarray(self.A, dtype=float)
            b_val = _resolve(b, self.b, "b")
            constraints.append(A @ x == np.broadcast_to(b_val, (A.shape[0],)))
        if self.C is not None:
            C = self.C.toarray() if spa.issparse(self.C) else np.asarray(self.C, dtype=float)
            d_val = _resolve(d, self.d, "d")
            constraints.append(C @ x <= np.broadcast_to(d_val, (C.shape[0],)))
        # Pin the conic solver (cvxpy's default may be an unavailable/expired backend).
        cp.Problem(cp.Minimize(objective), constraints).solve(solver=cp.CLARABEL)
        return np.asarray(x.value, dtype=float)
