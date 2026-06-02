"""Prompt construction for the coding-agent phases."""

from __future__ import annotations

from fomtools.codegen.ir import FomIR
from fomtools.codegen.targets import TargetDescriptor


def phase_a_instructions(ir: FomIR, target: TargetDescriptor) -> str:
    """Initial faithful-translation instructions."""
    verify = " ".join(target.verify_command())
    src = target.solver_source_path()
    return f"""\
You are generating a high-performance, embedded implementation of a verified
first-order optimization method named `{ir.name}`.

Write the file `{src}` ONLY. Everything else in this directory is already
generated and must not be changed:

- `include/fom_solver.h` defines the ABI you must implement exactly:
  `fom_set_default_settings`, `fom_setup`, `fom_update_params`, `fom_solve`,
  `fom_cleanup`.
- `include/fom_workspace.h` defines the workspace struct and dimension macros.
- `include/fom_data.h` / `src/fom_data.c` hold ALL numeric coefficients and
  activation constants (matrices `FOM_MAT_*`, `FOM_STATE_INIT`, soft-threshold
  `FOM_*_LAMBDA`, clip `FOM_*_LOWER/UPPER`). Do NOT hardcode numbers; read them
  from these symbols.
- The algorithm is specified in `test/SPEC.md`.

Hard constraints for the embedded target:
- No dynamic memory: no malloc/calloc/realloc/free, no variable-length arrays.
- Division-free: do not use the `/` operator anywhere in the code. The affine
  inverse has been folded into the coefficient matrices precisely so that no
  division is needed.
- Use only the statically-sized buffers already declared in the workspace.

Each solve must start from `FOM_STATE_INIT` (reset the state in `fom_setup` and
`fom_update_params`) so results match the reference exactly.

Run `{verify}`; it builds your code and checks it against the reference oracle
over many samples. Iterate until it prints `VERIFY OK` and exits 0. Strive for
the fastest correct implementation.
"""


def phase_b_instructions(ir: FomIR, target: TargetDescriptor) -> str:
    """Performance-optimization instructions."""
    verify = " ".join(target.verify_command())
    src = target.solver_source_path()
    return f"""\
The solver `{src}` already passes `{verify}`. Now optimize it for speed WITHOUT
breaking correctness:

- Hoist every iteration-invariant computation out of the main loop (precompute
  once in `fom_setup` / `fom_update_params` into workspace scratch).
- Exploit structure in the coefficient matrices (sparsity, symmetry, identity
  blocks) recorded in `test/spec.json`.
- Keep the embedded constraints: no dynamic allocation, division-free.

`{verify}` must still print `VERIFY OK`. If a `make bench` target exists, use it
to confirm you improved throughput.
"""
