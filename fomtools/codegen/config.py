"""Configuration objects for the FOMTools code generator.

These dataclasses describe *what* the user wants generated (target language,
embedded vs. large-scale, termination criterion, numerical dtype) without
touching the algorithm-specific intermediate representation, which lives in
`fomtools.codegen.ir`.
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

    ``OBJECTIVE`` stops early once the *verification objective* ``r^k`` -- the
    performance metric set with `Verifier.maximize`,
    e.g. a fixed-point or KKT residual, in the norm it was set with -- is at most
    ``tol``. The stopping test is then exactly the certified quantity: if
    verification shows ``r^K <= tol`` over the whole parameter set, the generated
    solver stops within ``K`` iterations. The norm follows the verifier's
    convention, so ``ord=2`` is the *squared* two-norm and ``tol`` bounds it.
    """

    MAX_ITERS = "max_iters"
    INF_NORM_CHANGE = "inf_norm_change"
    OBJECTIVE = "objective"


#: Default ``check_termination`` per criterion. The infinity-norm change is fused
#: into the state copy, so checking it every iteration is nearly free. The
#: verification objective costs its own matrix-vector products (a KKT residual
#: reads the problem data again), so it is checked every 25 iterations, as OSQP
#: does by default.
_CHECK_TERMINATION_DEFAULTS = {"objective": 25}
_TOL_KINDS = (TerminationKind.INF_NORM_CHANGE, TerminationKind.OBJECTIVE)


class TerminationSpec(BaseModel):
    """Termination criterion for the generated solve loop.

    Attributes
    ----------
    kind : TerminationKind
        Fixed iteration count, infinity-norm change, or the verification
        objective, each with early exit except the first.
    max_iters : int
        Hard upper bound on iterations. Always required: embedded targets size
        their (static) iteration accounting from it and every loop needs a guard.
    tol : float or None
        Tolerance for the ``INF_NORM_CHANGE`` and ``OBJECTIVE`` criteria. Ignored
        for ``MAX_ITERS``. Must be positive when the criterion uses it. For
        ``OBJECTIVE`` it bounds the verification objective as the verifier defines
        it, so with a 2-norm objective (``ord=2``) it bounds the SQUARED norm:
        ``tol=1e-6`` stops once ``||r||_2 <= 1e-3``, not ``1e-6``.
    check_termination : int
        How often the criterion is evaluated (the name and semantics follow
        OSQP's setting): ``N`` checks every ``N`` iterations (residual computed
        only then), ``0`` disables early exit. Ignored for ``MAX_ITERS``.

        For ``INF_NORM_CHANGE`` the default is 1 because the generated check is an
        infinity norm fused into the state copy (measured 1-2.5% of iteration
        time for direct-method FOMs), not extra residual matvecs; raise it for
        matrix-free methods on large problems where iterations are cheap. With
        ``0`` the loop runs ``max_iters`` (equivalent to ``MAX_ITERS`` but keeps
        the criterion recorded).

        For ``OBJECTIVE`` the default is 25: the objective costs its own
        matrix-vector products. The objective is also evaluated and tested at the
        last iteration (``k == max_iter``), whatever ``N`` is, so the reported
        residual is always the objective at the returned iterate and a solver run
        to a verified horizon reports ``FOM_SOLVED``. With ``0`` it is evaluated
        only there.
    """

    model_config = ConfigDict(frozen=True)

    kind: TerminationKind = TerminationKind.MAX_ITERS
    max_iters: int = Field(ge=1, default=1000, description="Hard upper bound on iterations.")
    tol: float | None = Field(
        default=None,
        description="Tolerance for the ``INF_NORM_CHANGE`` and ``OBJECTIVE`` criteria.",
    )
    check_termination: int = Field(
        ge=0,
        default=1,
        description=(
            "How often the termination criterion is evaluated (default 1 for "
            "``INF_NORM_CHANGE``, 25 for ``OBJECTIVE``)."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _default_check_termination(cls, data):
        # The default interval depends on the criterion, so fill it here rather
        # than in the field.
        if isinstance(data, dict) and data.get("check_termination") is None:
            data = dict(data)
            data.pop("check_termination", None)
            kind = TerminationKind(data.get("kind", TerminationKind.MAX_ITERS))
            if kind.value in _CHECK_TERMINATION_DEFAULTS:
                data["check_termination"] = _CHECK_TERMINATION_DEFAULTS[kind.value]
        return data

    @model_validator(mode="after")
    def _check_tol(self) -> "TerminationSpec":
        if self.kind in _TOL_KINDS and (self.tol is None or self.tol <= 0):
            raise ValueError(f"tol must be a positive float for the {self.kind.value} criterion")
        return self

    @property
    def uses_tolerance(self) -> bool:
        """Whether the loop checks the infinity-norm change criterion at all.

        Single source of truth shared by the emitted header (``FOM_USE_TOL``)
        and the reference oracle, so the generated code and the verification
        ground truth can never disagree on when early exit is possible.
        """
        return self.kind == TerminationKind.INF_NORM_CHANGE and self.check_termination > 0

    @property
    def uses_objective(self) -> bool:
        """Whether the loop evaluates the verification objective (``FOM_USE_OBJECTIVE``).

        True even with ``check_termination == 0``: the objective is then still
        evaluated, and tested, once at the last iteration.
        """
        return self.kind == TerminationKind.OBJECTIVE

    def is_check_iteration(self, k: int, max_iter: int | None = None) -> bool:
        """Whether the 1-based iteration ``k`` evaluates the ``OBJECTIVE`` criterion.

        Every ``check_termination``-th iteration, and the last one (``max_iter``,
        defaulting to ``max_iters``). Shared by the oracle and the emitted
        verify data so both test at the same iterations.
        """
        last = self.max_iters if max_iter is None else max_iter
        check = self.check_termination
        return k == last or (check > 0 and k % check == 0)


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
    num_verify_samples : int
        Number of random parameter instances drawn for the golden verify data.
    seed : int
        RNG seed for reproducible verify-sample generation.
    agent_timeout_s : int
        Wall-clock budget, in seconds, for the translation phase, shared by
        all of its agent attempts. Default 600.
    agent_optimize_max_turns : int
        Turn budget for one coding-agent attempt at the code optimization
        phase. Default 200.
    agent_optimize_timeout_s : int
        Wall-clock budget, in seconds, for the code optimization phase.
        Default 7200.
    cg_auto_min_dim : int
        Dimension threshold for the ``AUTO`` → CG upgrade on the large-scale
        CUDA backend (see `SolveMethod`). Raise it to keep more systems
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
    num_verify_samples: int = 8
    seed: int = 0
    agent_timeout_s: int = Field(
        ge=1,
        default=600,
        description=(
            "Wall-clock budget, in seconds, for the translation phase, shared by "
            "all of its agent attempts. The agent is told the time left after each "
            "tool call, a shell command cannot outlive it, and once it is spent the "
            "agent is stopped (after a short grace period to write its summary) "
            "and no further attempt starts. The phase is a bounded task, so the "
            "default is generous for it."
        ),
    )
    agent_optimize_max_turns: int = Field(
        ge=1,
        default=200,
        description=(
            "Turn budget for one coding-agent attempt at the code optimization phase. "
            "The phase is a profile/rewrite/verify/benchmark loop, so a method "
            "with more structure to exploit needs more turns; exhausting the "
            "budget leaves the verified solver in place but costs the agent's own "
            "summary of what it did. Sized so the wall-clock budget, not turns, "
            "is what ends a run at the default."
        ),
    )
    agent_optimize_timeout_s: int = Field(
        ge=1,
        default=7200,
        description=(
            "Wall-clock budget, in seconds, for the code optimization phase, "
            "enforced as for agent_timeout_s. Larger than the translation budget "
            "because the phase is open-ended: profile, rewrite, re-verify, "
            "benchmark, repeat, and each round of a GPU kernel rewrite (nvcc "
            "build, verify, benchmark) is slow. Two hours by default: a CUDA CG "
            "solve stopped at 30 minutes was still improving. A stopped agent "
            "still leaves whatever it had written on disk, which the pipeline "
            "re-verifies and benchmarks. The gates that run after the agent "
            "(verify, benchmark, profile) take seconds and are outside the budget."
        ),
    )
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
