"""Code generation for verified first-order methods.

Turns a FOM recorded in a :class:`fomtools.verifier.Verifier` into an
OSQP-codegen-style low-level solver (embedded C; large-scale CUDA/OpenMP) plus a
tiny Python wrapper. An LLM coding agent translates each step into the target
language and iterates against a Python reference oracle until the generated code
is numerically verified.

The public surface:

* :func:`extract_unrolled_ir` — the whole recorded run as one unrolled body,
  straight from the loop the user wrote to verify with. Takes arbitrary
  per-iteration coefficients and step types, and anchored methods. Rolls the run
  back up when repeating one iteration reproduces it.
* :func:`extract_ir` / :class:`FomIR` — the single-iteration intermediate
  representation extracted from a verifier.
* :func:`extract_scheduled_ir` / :class:`Recording` / :func:`as_schedule` — the
  same, for a method whose varying coefficients are *declared*, which buys a
  compact solver at a long horizon. The coefficients themselves are the caller's
  to supply; see :mod:`fomtools.codegen.schedule`.
* :class:`ReferenceOracle` — ground-truth NumPy execution of the FOM.
* :func:`check_stationary` / :func:`check_unrolled` — verify that the extracted
  IR reproduces the recorded tape. The first is what catches a method that is not
  stationary; the second gates every unrolled extraction.
* :class:`CodegenConfig` — what to generate (embedded/large-scale, backend,
  termination, dtype).
* :func:`generate_code` — the end-to-end entry point.
* :class:`AgentDriver` / :class:`StubAgentDriver` — the (mockable) coding-agent
  interface.
"""

from fomtools.codegen.agent import AgentDriver, AgentResult, AgentTask, StubAgentDriver
from fomtools.codegen.config import (
    Backend,
    CodegenConfig,
    SolveMethod,
    TerminationKind,
    TerminationSpec,
    Version,
)
from fomtools.codegen.emit import scaffold
from fomtools.codegen.extract import extract_ir
from fomtools.codegen.generate import CodegenResult, generate_code
from fomtools.codegen.harness import BenchResult, ProfRegion, TimingReport
from fomtools.codegen.ir import (
    FomIR,
    NonStationaryFOMError,
    ScheduleFitError,
    UnrolledBlowupError,
    UnrolledFidelityError,
)
from fomtools.codegen.oracle import (
    OracleResult,
    ReferenceOracle,
    check_stationary,
    check_unrolled,
    run_faithful,
    run_recorded_tape,
)
from fomtools.codegen.schedule import Schedule, as_schedule
from fomtools.codegen.extract_scheduled import Recording, extract_scheduled_ir
from fomtools.codegen.extract_unrolled import extract_unrolled_ir

__all__ = [
    "AgentDriver",
    "AgentResult",
    "AgentTask",
    "StubAgentDriver",
    "Backend",
    "CodegenConfig",
    "SolveMethod",
    "TerminationKind",
    "TerminationSpec",
    "Version",
    "scaffold",
    "extract_ir",
    "generate_code",
    "CodegenResult",
    "TimingReport",
    "BenchResult",
    "ProfRegion",
    "FomIR",
    "NonStationaryFOMError",
    "ScheduleFitError",
    "UnrolledBlowupError",
    "UnrolledFidelityError",
    "OracleResult",
    "ReferenceOracle",
    "run_faithful",
    "run_recorded_tape",
    "check_stationary",
    "check_unrolled",
    "Schedule",
    "as_schedule",
    "Recording",
    "extract_scheduled_ir",
    "extract_unrolled_ir",
]
