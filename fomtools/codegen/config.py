"""Configuration objects for the FOMTools code generator.

These dataclasses describe *what* the user wants generated (target language,
embedded vs. large-scale, termination criterion, numerical dtype) without
touching the algorithm-specific intermediate representation, which lives in
:mod:`fomtools.codegen.ir`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Version(str, Enum):
    """Which flavour of solver to emit.

    ``EMBEDDED`` mirrors OSQP's codegen: statically sized workspace, no dynamic
    memory allocation, and divisions avoided (factorizations / reciprocals are
    precomputed offline). ``LARGE_SCALE`` targets throughput on big problems via
    a multithreaded CPU (OpenMP) or GPU (CUDA) backend.
    """

    EMBEDDED = "embedded"
    LARGE_SCALE = "large_scale"


class Backend(str, Enum):
    """Target language / hardware for the emitted solver."""

    C = "c"
    CUDA = "cuda"
    OPENMP = "openmp"


class SolveMethod(str, Enum):
    """Strategy for the affine (linear-system) step. Inverses are never formed.

    ``AUTO`` picks a sparse *direct* factorization: LDL^T for symmetric systems
    (e.g. ADMM/OSQP KKT), LU otherwise. ``LDLT`` / ``LU`` force a direct method.
    ``CG`` (preconditioned conjugate gradient) is an *indirect* strategy for
    large / GPU problems where a factorization is too expensive: it never
    factorizes or inverts, reaching the solution through sparse matrix-vector
    products. It requires a symmetric positive-definite system and a few scalar
    divisions per iteration, so it is only available for the large-scale version
    (the embedded target is division-free).
    """

    AUTO = "auto"
    LDLT = "ldlt"
    LU = "lu"
    CG = "cg"


class TerminationKind(str, Enum):
    """How the generated solve loop decides to stop.

    ``MAX_ITERS`` runs a fixed iteration count (exactly matches the verifier /
    simulator semantics, so it is the most robust to verify). ``INF_NORM_CHANGE``
    additionally stops early once ``||x_k - x_{k-1}||_inf <= tol``.
    """

    MAX_ITERS = "max_iters"
    INF_NORM_CHANGE = "inf_norm_change"


@dataclass(frozen=True)
class TerminationSpec:
    """Termination criterion for the generated solve loop.

    Parameters
    ----------
    kind:
        Either fixed iteration count or infinity-norm change with early exit.
    max_iters:
        Hard upper bound on iterations. Always required: embedded targets size
        their (static) iteration accounting from it and every loop needs a guard.
    tol:
        Tolerance for the ``INF_NORM_CHANGE`` criterion. Ignored for
        ``MAX_ITERS``. Must be positive when the criterion uses it.
    check_termination:
        How often the ``INF_NORM_CHANGE`` criterion is evaluated (the name and
        semantics follow OSQP's setting): ``N`` checks every ``N`` iterations
        (residual computed only then), ``0`` disables checking entirely (the
        loop runs ``max_iters``; equivalent to ``MAX_ITERS`` but keeps the
        criterion recorded). The default is 1 because the generated check is an
        infinity norm fused into the state copy (measured 1-2.5% of iteration
        time for direct-method FOMs), not extra residual matvecs; raise it for
        matrix-free methods on large problems where iterations are cheap.
        Ignored for ``MAX_ITERS``.
    """

    kind: TerminationKind = TerminationKind.MAX_ITERS
    max_iters: int = 1000
    tol: float | None = None
    check_termination: int = 1

    def __post_init__(self) -> None:
        if self.max_iters <= 0:
            raise ValueError(f"max_iters must be positive, got {self.max_iters}")
        if self.check_termination < 0:
            raise ValueError(
                f"check_termination must be >= 0 (0 disables checking), "
                f"got {self.check_termination}"
            )
        if self.kind == TerminationKind.INF_NORM_CHANGE:
            if self.tol is None or self.tol <= 0:
                raise ValueError("tol must be a positive float for the inf_norm_change criterion")

    @property
    def uses_tolerance(self) -> bool:
        """Whether the loop checks the tolerance criterion at all.

        Single source of truth shared by the emitted header (``FOM_USE_TOL``)
        and the reference oracle, so the generated code and the verification
        ground truth can never disagree on when early exit is possible.
        """
        return self.kind == TerminationKind.INF_NORM_CHANGE and self.check_termination > 0


@dataclass(frozen=True)
class CodegenConfig:
    """Top-level knobs controlling a code-generation run.

    Parameters
    ----------
    version:
        Embedded (no malloc / division-light) or large-scale.
    backend:
        Target language. ``CUDA`` / ``OPENMP`` are only meaningful with
        ``version == LARGE_SCALE``; ``C`` pairs with ``EMBEDDED``.
    termination:
        Loop termination criterion.
    dtype:
        Floating-point type emitted in the C ABI (``"float64"`` or
        ``"float32"``). Large-scale GPU backends commonly prefer ``float32``.
    tol_verify:
        Tolerance used by the equivalence harness when comparing generated-code
        output against the Python reference oracle.
    allow_unroll:
        When extraction cannot roll the recorded steps into a stationary loop
        body (e.g. iteration-dependent step sizes), emit a straight-line,
        loop-free body instead of raising.
    num_verify_samples:
        Number of random parameter instances drawn for the golden verify data.
    seed:
        RNG seed for reproducible verify-sample generation.
    """

    version: Version = Version.EMBEDDED
    backend: Backend = Backend.C
    termination: TerminationSpec = field(default_factory=TerminationSpec)
    solve_method: SolveMethod = SolveMethod.AUTO
    dtype: str = "float64"
    tol_verify: float = 1e-6
    allow_unroll: bool = False
    num_verify_samples: int = 8
    seed: int = 0

    def __post_init__(self) -> None:
        if self.dtype not in ("float64", "float32"):
            raise ValueError(f"dtype must be 'float64' or 'float32', got {self.dtype!r}")
        if self.version == Version.EMBEDDED and self.backend != Backend.C:
            raise ValueError(
                f"embedded version only supports the C backend; got backend={self.backend.value!r}"
            )
        if self.solve_method == SolveMethod.CG and self.version == Version.EMBEDDED:
            raise ValueError(
                "the CG (indirect) solve method needs runtime division and is only available "
                "for the large-scale version; the embedded target is division-free. Use "
                "solve_method=AUTO/LDLT/LU for the embedded version."
            )
        if self.tol_verify <= 0:
            raise ValueError("tol_verify must be positive")

    @property
    def c_float(self) -> str:
        """C type for the emitted ``fom_float`` typedef."""
        return "double" if self.dtype == "float64" else "float"
