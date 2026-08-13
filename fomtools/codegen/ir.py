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


class ActivationKind(str, Enum):
    """Elementwise nonlinearity applied after a step's linear op."""

    IDENTITY = "identity"  # affine / linear recombination
    SOFT_THRESHOLD = "soft_threshold"
    RELU = "relu"
    SATURATED_LINEAR = "saturated_linear"


@dataclass(frozen=True)
class MatrixData:
    """A named numeric coefficient block (matrix) or vector (const).

    Attributes
    ----------
    name:
        Stable symbolic identifier used by both the IR and the emitted code
        (e.g. ``"A0"``, ``"const0"``).
    array:
        The dense values. Sparse inputs are densified here for the embedded
        backend; :attr:`sparse_pattern` records the original structure when it
        matters for large-scale spMV emission.
    sparse_pattern:
        Optional ``(rows, cols)`` index arrays of the structurally non-zero
        entries, retained from a sparse source for downstream spMV codegen.
    """

    name: str
    array: np.ndarray
    sparse_pattern: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    @classmethod
    def from_coeff(cls, name: str, coeff: np.ndarray | spa.spmatrix) -> "MatrixData":
        """Build from a (possibly sparse) coefficient, densifying for storage
        while preserving the sparsity pattern."""
        if spa.issparse(coeff):
            coo = coeff.tocoo()
            dense = np.asarray(coeff.todense(), dtype=float)
            return cls(name=name, array=dense, sparse_pattern=(coo.row, coo.col))
        return cls(name=name, array=np.atleast_2d(np.asarray(coeff, dtype=float)))


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
        iteration counter. Empty for a fully stationary method.
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
