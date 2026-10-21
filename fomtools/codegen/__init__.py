"""Code generation for verified first-order methods.

Turns a FOM recorded in a `fomtools.verifier.Verifier` into an
OSQP-codegen-style low-level solver (embedded C; large-scale CUDA/OpenMP) plus a
tiny Python wrapper. An LLM coding agent translates each step into the target
language and iterates against a Python reference oracle until the generated code
is numerically verified.

The public surface:

* `extract_unrolled_ir` — the whole recorded run as one unrolled body,
  straight from the loop the user wrote to verify with. Takes arbitrary
  per-iteration coefficients and step types, and anchored methods. Rolls the run
  back up when repeating one iteration reproduces it.
* `extract_ir` / `FomIR` — the single-iteration intermediate
  representation extracted from a verifier.
* `extract_scheduled_ir` / `Recording` / `as_schedule` — the
  same, for a method whose varying coefficients are *declared*, which buys a
  compact solver at a long horizon. The coefficients themselves are the caller's
  to supply; see `fomtools.codegen.schedule`.
* `ReferenceOracle` — ground-truth NumPy execution of the FOM.
* `check_stationary` / `check_unrolled` — verify that the extracted
  IR reproduces the recorded tape. The first is what catches a method that is not
  stationary; the second gates every unrolled extraction.
* `CodegenConfig` — what to generate (embedded/large-scale, backend,
  termination, dtype). ``TerminationKind.OBJECTIVE`` stops the generated solver
  on the verification objective set with ``Verifier.maximize``; see
  `fomtools.codegen.objective` for how it is mapped onto the loop and
  `ObjectiveExtractionError` for what cannot be.
* `generate_code` — the end-to-end entry point.
* `AgentDriver` / `StubAgentDriver` — the (mockable) coding-agent
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
from fomtools.codegen.objective import ObjectiveExtractionError
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
    "ObjectiveExtractionError",
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
