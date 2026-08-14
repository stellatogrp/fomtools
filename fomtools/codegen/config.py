"""Configuration objects for the FOMTools code generator.

These dataclasses describe *what* the user wants generated (target language,
embedded vs. large-scale, termination criterion, numerical dtype) without
touching the algorithm-specific intermediate representation, which lives in
:mod:`fomtools.codegen.ir`.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Verify-tolerance default per dtype. Single precision (~1e-7 relative error)
#: needs a looser equivalence gate than double; see ``CodegenConfig.tol_verify``.
_TOL_VERIFY_DEFAULTS = {"float64": 1e-6, "float32": 1e-4}


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

    On the large-scale **CUDA** backend, ``AUTO`` additionally *upgrades* to CG
    for systems that are large (dimension ``>= cg_auto_min_dim``) and SPD,
    because CG is matrix-free and fully data-parallel — the GPU-friendly choice
    — whereas the direct triangular solves are sequential and run on the host.
    Small SPD systems and all non-SPD systems (including quasidefinite KKT,
    which is symmetric-indefinite) keep AUTO's direct factorization. Forcing
    ``LDLT`` / ``LU`` / ``CG`` always bypasses this heuristic.
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


class TerminationSpec(BaseModel):
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

    model_config = ConfigDict(frozen=True)

    kind: TerminationKind = TerminationKind.MAX_ITERS
    max_iters: int = Field(ge=1, default=1000, description="Hard upper bound on iterations.")
    tol: float | None = Field(
        default=None, description="Tolerance for the ``INF_NORM_CHANGE`` criterion."
    )
    check_termination: int = Field(
        ge=0, default=1, description="How often the ``INF_NORM_CHANGE`` criterion is evaluated."
    )

    @model_validator(mode="after")
    def _check_tol(self) -> "TerminationSpec":
        if self.kind == TerminationKind.INF_NORM_CHANGE and (self.tol is None or self.tol <= 0):
            raise ValueError("tol must be a positive float for the inf_norm_change criterion")
        return self

    @property
    def uses_tolerance(self) -> bool:
        """Whether the loop checks the tolerance criterion at all.

        Single source of truth shared by the emitted header (``FOM_USE_TOL``)
        and the reference oracle, so the generated code and the verification
        ground truth can never disagree on when early exit is possible.
        """
        return self.kind == TerminationKind.INF_NORM_CHANGE and self.check_termination > 0


class CodegenConfig(BaseModel):
    """Top-level knobs controlling a code-generation run.

    Attributes
    ----------
    version : Version
        Embedded (no malloc / division-light) or large-scale.
    backend : Backend
        Target language. ``CUDA`` / ``OPENMP`` are only meaningful with
        ``version == LARGE_SCALE``; ``C`` pairs with ``EMBEDDED``.
    termination : TerminationSpec
        Loop termination criterion.
    solve_method : SolveMethod
        Linear-system solution strategy. ``AUTO`` selects an appropriate
        direct or iterative method from the version, backend, and problem size.
    dtype : {"float64", "float32"}
        Floating-point type emitted in the C ABI (``"float64"`` or
        ``"float32"``). Large-scale GPU backends commonly prefer ``float32``.
    tol_verify : float or None
        Tolerance used by the equivalence harness when comparing generated-code
        output against the Python reference oracle. When left unset it defaults
        by dtype (1e-6 for float64, 1e-4 for float32, since single precision
        cannot meet a 1e-6 gate); set it explicitly to override.
    allow_unroll : bool
        When extraction cannot roll the recorded steps into a stationary loop
        body (e.g. iteration-dependent step sizes), emit a straight-line,
        loop-free body instead of raising.
    num_verify_samples : int
        Number of random parameter instances drawn for the golden verify data.
    seed : int
        RNG seed for reproducible verify-sample generation.
    cg_auto_min_dim : int
        Dimension threshold for the ``AUTO`` → CG upgrade on the large-scale
        CUDA backend (see :class:`SolveMethod`). Raise it to keep more systems
        on a direct factorization; lower it to prefer CG for smaller SPD
        systems. Ignored for non-CUDA backends and explicit solve methods.
    """

    version: Version = Version.EMBEDDED
    backend: Backend = Backend.C
    termination: TerminationSpec = Field(default_factory=TerminationSpec)
    solve_method: SolveMethod = SolveMethod.AUTO
    dtype: Literal["float64", "float32"] = "float64"
    tol_verify: float | None = Field(
        default=None,
        description=(
            "Tolerance used by the equivalence harness when comparing generated "
            "code against the reference oracle. Defaults by dtype when left unset: "
            "1e-6 for float64, 1e-4 for float32 (single precision carries ~1e-7 "
            "relative error, which 1e-6 is too tight to absorb — especially for "
            "iterative CG). Set explicitly to override."
        ),
    )
    allow_unroll: bool = False
    num_verify_samples: int = 8
    seed: int = 0
    cg_auto_min_dim: int = Field(
        ge=1,
        default=256,
        description=(
            "Minimum affine-system dimension at which solve_method=AUTO upgrades "
            "a large-scale CUDA SPD solve to CG (matrix-free, data-parallel). "
            "Smaller SPD systems and all non-SPD systems keep a direct "
            "factorization. No effect for non-CUDA backends or an explicit "
            "solve_method."
        ),
    )

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="before")
    @classmethod
    def _default_tol_verify(cls, data):
        # Fill the dtype-dependent tol_verify default when the user left it unset.
        if isinstance(data, dict) and data.get("tol_verify") is None:
            dtype = data.get("dtype", "float64")
            data["tol_verify"] = _TOL_VERIFY_DEFAULTS.get(dtype, 1e-6)
        return data

    @model_validator(mode="after")
    def _check_tol_verify(self) -> "CodegenConfig":
        if self.tol_verify is None or self.tol_verify <= 0:
            raise ValueError("tol_verify must be a positive float")
        return self

    @model_validator(mode="after")
    def _check_version_constraints(self) -> "CodegenConfig":
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
        return self

    @property
    def c_float(self) -> str:
        """C type for the emitted ``fom_float`` typedef."""
        return "double" if self.dtype == "float64" else "float"
