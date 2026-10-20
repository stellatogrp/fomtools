# Lasso Code Generation Speedup

This tutorial benchmarks iterative soft-thresholding for a small Lasso problem.
It compares a direct NumPy implementation with a verified generated C solver,
then reports the speedup from code generation and the subsequent C optimization
pass.

Open the companion [marimo notebook](lasso.html), or edit the source locally
from the repository root:

```bash
uv run marimo edit docs/tutorials/codegen/lasso.py
```

## Problem

The Lasso instance is

$$
\begin{array}{ll}
    \text{minimize} & (1/2)\|Dz - x\|_2^2 + \lambda \|z\|_1,
\end{array}
$$

where $D \in \mathbf{R}^{p \times n}$, $z \in \mathbf{R}^n$, and $x$
is the parameter.

## Method

The solver uses iterative soft-thresholding:

$$
z^{k+1} =
\mathbf{soft}_{\lambda\eta}\left(z^k - \eta D^T(Dz^k - x)\right).
$$

Both the NumPy reference and the generated C solver run the same update rule for
the same number of iterations.

## Speedup

The recorded benchmark ran 200 ISTA iterations per solve.

| implementation | time per solve |
|---|---:|
| direct NumPy Python | `1053.063 us` |
| generated C after code generation | `7.335 us` |
| generated C after C optimization | `1.546 us` |

The generated C solver is `143.6x` faster than direct Python. The C
optimization pass adds another `4.75x` speedup over the initial generated C
solver, for a total speedup of `681.3x` from direct Python to optimized C. The
optimized solver was kept after verification with worst max-absolute error
`0.000e+00`.

## Optimization Trick

The main trick for Lasso is scalarizing the tiny fixed-size ISTA loop. The
optimized solver replaces generic matrix and state-update helpers with an
explicit four-variable recurrence, hoists the parameter-dependent affine terms
once per solve, applies soft-thresholding with direct scalar code, and keeps the
four state variables in local scalars during the hot loop. The workspace arrays
are written back only when the final state is needed.

In this example, the bottleneck was mostly helper-call, indexing, and loop
overhead rather than arithmetic.
