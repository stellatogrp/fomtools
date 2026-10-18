"""Intermediate representation (IR) for a verified first-order method.

The IR captures **one iteration** of a stationary FOM as a small straight-line
program over named vectors:

* a fixed-size *state* vector (the loop-carried iterate),
* zero or more *parameter* vectors (problem data, supplied at solve time),
* a sequence of *body steps*, each computing a fresh *temporary* vector as
  ``activation(linear_op(state, params, earlier temporaries))``, and
* a *state update* linear op producing the next state from the current state
  and the temporaries.

Every step in FOMTools (:class:`~fomtools.steps.affine.Affine`,
:class:`~fomtools.steps.relu.Relu`,
:class:`~fomtools.steps.soft_threshold.SoftThreshold`,
:class:`~fomtools.steps.saturated_linear.SaturatedLinear`) lowers to a
:class:`LinearOp` followed by an elementwise :class:`ActivationOp`.  In
particular the affine operator's matrix inverse is *folded into* the linear-op
coefficients (see :attr:`fomtools.steps.affine.Affine.lhs_mat_inverse`), so the
affine activation is the identity and no division survives into the emitted
code.

The numeric matrices/constants referenced by name live in :class:`MatrixData`
objects collected on :class:`FomIR`; the IR proper is light and serializable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import scipy.sparse as spa

from fomtools.codegen.config import TerminationSpec
from fomtools.codegen.factorize import Factorization


class NonStationaryFOMError(ValueError):
    """Raised when the recorded steps cannot be rolled into a single repeating
    iteration body.

    A coefficient that varies with the iteration index is *not* in itself a
    reason to raise: declare it as a :mod:`~fomtools.codegen.schedule` and it
    lowers to a table read at the loop counter. This error means the recording
    cannot be rolled at all — the body changes shape between iterations, or a
    varying quantity reaches somewhere a table cannot (see
    :class:`ScheduleFitError`).
    """


class ScheduleFitError(NonStationaryFOMError):
    """Raised when a declared schedule does not explain how a coefficient varies.

    Extraction probes the recording at several values of each schedule scalar and
    recovers a base/slope pair per coefficient block, assuming the dependence is
    affine (which it is whenever the scalar enters the recorded
    ``LinearExpression`` linearly). This error means the recovered fit failed to
    predict a held-out probe, so the emitted table would not reproduce the
    method.
    """


class UnrolledBlowupError(NonStationaryFOMError):
    """Raised when an unrolled recording would emit a quadratic number of terms.

    Unrolled extraction expresses each iterate in terms of leaves and earlier
    temporaries. When the recorded loop carries its iterate through a *non-leaf*
    expression -- ``z = z + ut - u``, as in Douglas-Rachford and ADMM -- every
    block reaches back through all earlier temporaries and the operation count
    grows with the square of the iteration count. Materializing each iterate as
    its own variable restores linear growth; see
    :class:`fomtools.iteratetrajectory.IterateTrajectory`.
    """


class UnrolledFidelityError(NonStationaryFOMError):
    """Raised when an unrolled IR does not reproduce the recorded tape.

    Unlike :class:`NonStationaryFOMError`, this is not something a recording can
    provoke by being iteration-varying: an unrolled body gives every iteration
    its own coefficients, so it should reproduce the tape exactly whatever the
    method does. Raising means extraction and the tape disagree.
    """


class ActivationKind(str, Enum):
    """Elementwise nonlinearity applied after a step's linear op."""

    IDENTITY = "identity"  # affine / linear recombination
    SOFT_THRESHOLD = "soft_threshold"
    RELU = "relu"
    SATURATED_LINEAR = "saturated_linear"


#: Magnitude below which a stored coefficient is treated as structurally zero
#: when emitting it (CSR arrays, nonzero counts).
EMIT_ZERO_TOL = 1e-12


@dataclass(frozen=True)
class MatrixData:
    """A named numeric coefficient block (matrix) or vector (const).

    Attributes
    ----------
    name:
        Stable symbolic identifier used by both the IR and the emitted code
        (e.g. ``"A0"``, ``"const0"``).
    array:
        The values: a 1-D ``ndarray`` for vectors (constants, clip bounds), and a
        2-D matrix for coefficient blocks, held as a SciPy CSR matrix when it was
        extracted sparse. Coefficient blocks of large problems are almost all
        zeros, so they are never densified; use :meth:`dense` and :meth:`csr`
        where a specific form is needed.
    sparse_pattern:
        Optional ``(rows, cols)`` index arrays of the structurally non-zero
        entries, retained from a sparse source for downstream spMV codegen.
    """

    name: str
    array: np.ndarray | spa.csr_matrix
    sparse_pattern: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    @property
    def is_sparse(self) -> bool:
        return spa.issparse(self.array)

    def dense(self) -> np.ndarray:
        """The values as a dense ``ndarray`` (only for small blocks)."""
        return self.array.toarray() if self.is_sparse else np.asarray(self.array, dtype=float)

    def csr(self, tol: float = EMIT_ZERO_TOL) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(rowptr, colidx, vals)`` of the entries larger than ``tol`` in magnitude."""
        return csr_arrays(self.array, tol)

    def nnz(self, tol: float = EMIT_ZERO_TOL) -> int:
        """Number of entries larger than ``tol`` in magnitude."""
        values = self.array.data if self.is_sparse else np.asarray(self.array)
        return int(np.count_nonzero(np.abs(values) > tol))

    @classmethod
    def from_coeff(cls, name: str, coeff: np.ndarray | spa.spmatrix) -> "MatrixData":
        """Build from a (possibly sparse) coefficient, stored as CSR either way so
        every coefficient block has one storage (and one rounding in the oracle);
        a sparse input also records its pattern."""
        if spa.issparse(coeff):
            m = spa.csr_matrix(coeff, dtype=float)
            coo = m.tocoo()
            return cls(name=name, array=m, sparse_pattern=(coo.row, coo.col))
        return cls(name=name, array=spa.csr_matrix(np.atleast_2d(np.asarray(coeff, dtype=float))))


def csr_arrays(matrix, tol: float = EMIT_ZERO_TOL) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(rowptr, colidx, vals)`` of a dense or sparse matrix's entries larger than
    ``tol`` in magnitude, rows in order and columns sorted within each row."""
    if spa.issparse(matrix):
        m = spa.csr_matrix(matrix, dtype=float, copy=True)
    else:
        m = spa.csr_matrix(np.atleast_2d(np.asarray(matrix, dtype=float)))
    m.data[np.abs(m.data) <= tol] = 0.0
    m.eliminate_zeros()
    m.sort_indices()
    return m.indptr, m.indices, m.data


def max_abs(values) -> float:
    """Largest magnitude of a dense or sparse array (0 for an empty one)."""
    data = values.data if spa.issparse(values) else np.asarray(values)
    return float(np.max(np.abs(data))) if data.size else 0.0


@dataclass(frozen=True)
class Term:
    """One ``matrix @ source`` summand of a :class:`LinearOp`.

    ``source`` is one of ``"state"``, ``"param:<name>"`` or ``"tmp:<name>"``.

    When :attr:`scale` is set, the summand is multiplied by that schedule's value
    at the current iteration, i.e. ``schedules[scale][k] * (matrix @ source)``. A
    coefficient block that varies with ``k`` is emitted as two terms: an unscaled
    base and a scaled slope.
    """

    source: str
    matrix: str  # name into FomIR.matrices
    scale: str | None = None  # name into FomIR.schedules, or None for a fixed term


@dataclass(frozen=True)
class LinearOp:
    """An affine map ``out = const + sum_i (matrix_i @ source_i)``.

    Attributes
    ----------
    out_dim:
        Dimension of the produced vector.
    terms:
        The ``matrix @ source`` summands.
    const:
        Name of the constant vector in ``FomIR.matrices``, or ``None`` for a
        zero constant.
    """

    out_dim: int
    terms: tuple[Term, ...]
    const: str | None = None


@dataclass(frozen=True)
class ActivationOp:
    """An elementwise nonlinearity.

    Attributes
    ----------
    kind:
        Which nonlinearity.
    lambd:
        Threshold for :data:`ActivationKind.SOFT_THRESHOLD`.
    lower / upper:
        Names of clip-bound vectors (in ``FomIR.matrices``) for
        :data:`ActivationKind.SATURATED_LINEAR`, or ``None`` when the bounds are
        scalars carried in :attr:`lower_scalar` / :attr:`upper_scalar`.
    lower_scalar / upper_scalar:
        Scalar clip bounds when the saturated-linear limits are constants.
    """

    kind: ActivationKind
    lambd: float | None = None
    lower: str | None = None
    upper: str | None = None
    lower_scalar: float | None = None
    upper_scalar: float | None = None


@dataclass(frozen=True)
class IterationStep:
    """One step of the iteration body.

    Normally ``tmp = activation(linear_op(...))``. For an affine operator the
    step instead *solves* a linear system: ``tmp = M^{-1} linear_op(...)`` via the
    factorization named by :attr:`solve` (the activation is the identity). The
    matrix inverse is never formed — see :mod:`fomtools.codegen.factorize`.
    """

    name: str  # temporary name, e.g. "t0"
    out_dim: int
    linear_op: LinearOp
    activation: ActivationOp
    solve: str | None = None  # name into FomIR.factorizations, or None


#: The norms an objective can be measured in, as the verifier defines them:
#: ``"inf"`` is the max of absolute values, ``"1"`` their sum, and ``"2"`` the
#: sum of squares (the verifier's two-norm objective is the *squared* norm).
OBJECTIVE_NORMS = ("1", "2", "inf")


@dataclass(frozen=True)
class ObjectiveBlock:
    """Rows ``activation(linear_op(...))`` of the verification objective ``r^k``.

    Evaluated at the end of iteration ``k``, once the new state exists and
    before it replaces the current one, so ``state`` holds ``w^{k-1}``,
    ``new_state`` holds ``w^k`` and ``tmp:<name>`` is iteration ``k``'s body
    temporary. Besides those the linear op may read ``param:<name>``.
    The activation is the identity or a ReLU (for one-sided terms such as
    ``max(Ax - u, 0)``).
    """

    name: str
    out_dim: int
    linear_op: LinearOp
    activation: ActivationOp


@dataclass(frozen=True)
class ObjectiveSpec:
    """The verification objective ``r^k = ||concat(blocks)||``, for early exit.

    Attributes
    ----------
    blocks:
        The rows of ``r^k``, grouped into blocks that each share one
        activation. The norm is taken over all of them at once, so their order
        does not matter.
    norm:
        One of :data:`OBJECTIVE_NORMS`. ``"2"`` is the sum of squares, matching
        :meth:`~fomtools.verifier.Verifier.set_objective_norm`.
    """

    blocks: tuple[ObjectiveBlock, ...]
    norm: str

    def __post_init__(self):
        if self.norm not in OBJECTIVE_NORMS:
            raise ValueError(f"objective norm must be one of {OBJECTIVE_NORMS}, got {self.norm!r}")

    @property
    def out_dim(self) -> int:
        return sum(block.out_dim for block in self.blocks)

    def linops(self) -> list[LinearOp]:
        return [block.linear_op for block in self.blocks]


@dataclass(frozen=True)
class ParamSpec:
    """A solve-time parameter vector."""

    name: str
    dim: int
    lb: np.ndarray
    ub: np.ndarray


@dataclass(frozen=True)
class StateSpec:
    """The loop-carried state vector.

    Attributes
    ----------
    dim:
        Total state dimension.
    slots:
        Ordered ``(name, dim)`` layout of the leaves concatenated into the state
        vector.
    init:
        Initial state values (the FOM's starting iterate ``z0``).
    """

    dim: int
    slots: tuple[tuple[str, int], ...]
    init: np.ndarray


@dataclass(frozen=True)
class FomIR:
    """Complete IR for one repeatable iteration, or for a whole unrolled run.

    Attributes
    ----------
    params:
        Solve-time parameter vectors.
    state:
        Loop-carried state layout + initial value. For an unrolled IR this is
        the solver's *input*: the initial iterates, plus any earlier iterate the
        method keeps referring to (a Halpern anchor, say).
    body:
        Ordered iteration-body steps producing temporaries. For an unrolled IR
        this is the whole run as one straight-line program, in dependency order.
    state_update:
        Linear op producing the next state from current state + temporaries. For
        an unrolled IR it produces the final answer.
    termination:
        Loop termination criterion.
    matrices:
        All named coefficient/const blocks referenced by the ops.
    schedules:
        Iteration-varying scalars referenced by :attr:`Term.scale`, each a table
        of at least ``termination.max_iters`` values indexed by the 0-based
        iteration counter. Empty for a fully stationary method. Mutually
        exclusive with :attr:`unrolled`: both express iteration-varying
        coefficients, and they do it in incompatible ways.
    dtype:
        Numeric dtype string (``"float64"`` / ``"float32"``).
    name:
        Human-friendly FOM name (used for fixtures / file naming).
    unrolled:
        Whether :attr:`body` is a whole run rather than one repeatable
        iteration. An unrolled body is executed **once**, so
        ``termination.max_iters`` is 1 and no residual-based early exit applies;
        each iteration carries its own coefficient blocks, which is what lets a
        recording use arbitrary per-iteration steps.
    objective:
        The verification objective the loop stops on under
        :attr:`~fomtools.codegen.config.TerminationKind.OBJECTIVE`, or ``None``.
        For an unrolled IR it is evaluated once, after the single pass.
    """

    params: tuple[ParamSpec, ...]
    state: StateSpec
    body: tuple[IterationStep, ...]
    state_update: LinearOp
    termination: TerminationSpec
    matrices: dict[str, MatrixData]
    factorizations: dict[str, Factorization] = field(default_factory=dict)
    schedules: dict[str, np.ndarray] = field(default_factory=dict)
    dtype: str = "float64"
    name: str = "fom"
    unrolled: bool = False
    objective: ObjectiveSpec | None = None

    def __post_init__(self):
        # Enforced here rather than left to "whichever extraction function you
        # happened to call": the two representations answer the same question
        # differently, and an IR holding both would make the emitted spec
        # self-contradictory -- "executed once, ignore max_iter" next to a table
        # indexed by a loop counter that no longer exists.
        if self.unrolled and self.schedules:
            raise ValueError(
                "an IR cannot be both unrolled and scheduled: an unrolled body already "
                "carries its own coefficients per iteration, so the schedule(s) "
                f"{sorted(self.schedules)} would be tabulated over a loop that is not "
                "emitted. Extract the run with extract_unrolled_ir (arbitrary variation, "
                "size grows with the horizon) or declare the variation with "
                "extract_scheduled_ir (compact, affine dependence only)."
            )
        if self.termination.uses_objective and self.objective is None:
            raise ValueError(
                "termination kind 'objective' needs the verification objective in the IR; "
                "set it on the verifier with set_objective_norm() before extracting."
            )

    @property
    def state_dim(self) -> int:
        return self.state.dim

    @property
    def is_scheduled(self) -> bool:
        """Whether any coefficient varies with the iteration index."""
        return bool(self.schedules)

    @property
    def max_rhs_dim(self) -> int:
        """Largest intermediate vector size; used to statically size scratch."""
        dims = [self.state.dim, self.state_update.out_dim]
        dims += [step.out_dim for step in self.body]
        dims += [step.linear_op.out_dim for step in self.body]
        return max(dims)

    def tmp_dims(self) -> dict[str, int]:
        """Map each temporary name to its dimension."""
        return {step.name: step.out_dim for step in self.body}

    def objective_inputs(self) -> list[str]:
        """The parameter and temporary sources the objective reads, in the order
        ``fom_objective`` takes them after ``(it, state, new_state)``: parameters in
        declaration order, then temporaries in body order."""
        if self.objective is None:
            return []
        used = {t.source for lin in self.objective.linops() for t in lin.terms}
        params = [f"param:{p.name}" for p in self.params if f"param:{p.name}" in used]
        tmps = [f"tmp:{s.name}" for s in self.body if f"tmp:{s.name}" in used]
        return params + tmps

    def linops(self) -> list[LinearOp]:
        """Every linear op: the body steps', the state update, and the objective's."""
        ops = [step.linear_op for step in self.body] + [self.state_update]
        if self.objective is not None:
            ops += self.objective.linops()
        return ops
