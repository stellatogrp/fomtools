"""Code generation for verified first-order methods.

Turns a FOM recorded in a :class:`fomtools.verifier.Verifier` into an
OSQP-codegen-style low-level solver (embedded C; large-scale CUDA/OpenMP) plus a
tiny Python wrapper. An LLM coding agent translates each step into the target
language and iterates against a Python reference oracle until the generated code
is numerically verified.

The public surface:

* :func:`extract_ir` / :class:`FomIR` — the single-iteration intermediate
  representation extracted from a verifier.
* :func:`extract_scheduled_ir` / :class:`Recording` — the same, for a method with
  an iteration-varying coefficient such as FISTA's momentum. See
  :mod:`fomtools.codegen.schedule`.
* :class:`ReferenceOracle` — ground-truth NumPy execution of the FOM.
* :func:`check_stationary` — verify that the rolled IR reproduces the recorded
  tape, which is what catches a method that is not stationary.
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
    TerminationKind,
    TerminationSpec,
    Version,
)
from fomtools.codegen.emit import scaffold
from fomtools.codegen.extract import extract_ir
from fomtools.codegen.generate import CodegenResult, generate_code
from fomtools.codegen.harness import BenchResult, ProfRegion, TimingReport
from fomtools.codegen.ir import FomIR, NonStationaryFOMError, ScheduleFitError
from fomtools.codegen.oracle import (
    OracleResult,
    ReferenceOracle,
    check_stationary,
    run_faithful,
    run_recorded_tape,
)
from fomtools.codegen.schedule import Schedule, fista_momentum, nesterov_momentum
from fomtools.codegen.scheduled import Recording, extract_scheduled_ir

__all__ = [
    "AgentDriver",
    "AgentResult",
    "AgentTask",
    "StubAgentDriver",
    "Backend",
    "CodegenConfig",
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
    "OracleResult",
    "ReferenceOracle",
    "run_faithful",
    "run_recorded_tape",
    "check_stationary",
    "Schedule",
    "nesterov_momentum",
    "fista_momentum",
    "Recording",
    "extract_scheduled_ir",
]
