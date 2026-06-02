"""Code generation for verified first-order methods.

Turns a FOM recorded in a :class:`fomtools.verifier.Verifier` into an
OSQP-codegen-style low-level solver (embedded C; large-scale CUDA/OpenMP) plus a
tiny Python wrapper. An LLM coding agent translates each step into the target
language and iterates against a Python reference oracle until the generated code
is numerically verified.

The public surface:

* :func:`extract_ir` / :class:`FomIR` — the single-iteration intermediate
  representation extracted from a verifier.
* :class:`ReferenceOracle` — ground-truth NumPy execution of the FOM.
* :class:`CodegenConfig` — what to generate (embedded/large-scale, backend,
  termination, dtype).
* :func:`generate_code` — the end-to-end entry point.
* :class:`AgentDriver` / :class:`StubAgentDriver` — the (mockable) coding-agent
  interface.
"""

from fomtools.codegen.config import (
    Backend,
    CodegenConfig,
    TerminationKind,
    TerminationSpec,
    Version,
)
from fomtools.codegen.extract import extract_ir
from fomtools.codegen.ir import FomIR, NonStationaryFOMError
from fomtools.codegen.oracle import OracleResult, ReferenceOracle, run_faithful

__all__ = [
    "Backend",
    "CodegenConfig",
    "TerminationKind",
    "TerminationSpec",
    "Version",
    "extract_ir",
    "FomIR",
    "NonStationaryFOMError",
    "OracleResult",
    "ReferenceOracle",
    "run_faithful",
]
