# Portfolio Code Generation Speedup

This tutorial benchmarks a small mean-variance portfolio problem. It compares a
direct NumPy implementation with a verified generated C solver, then reports the
speedup from code generation and the subsequent C optimization pass.

Open the companion [marimo notebook](portfolio.html), or edit the source locally
from the repository root:

```bash
uv run marimo edit docs/tutorials/portfolio.py
```

## Problem

The portfolio weights solve a constrained mean-variance problem,

$$
\begin{array}{ll}
    \text{minimize} & (1/2)x^\top \Sigma x - r^\top x \\
    \text{subject to} & 0 \le x \le u,\quad \mathbf{1}^\top x = 1.
\end{array}
$$

The notebook generates a positive definite covariance matrix, expected returns,
box bounds, and a budget equality.

## Method

The solver uses the alternating direction method of multipliers. Each iteration
clips an auxiliary variable to the constraint box, solves a small fixed linear
system, and updates the auxiliary variable. Both the NumPy reference and the
generated C solver run the same update rule for the same number of iterations.

With \(A = [I;\mathbf{1}^\top]\), lower and upper bounds \(\ell,u\), covariance
matrix \(\Sigma\), expected return vector \(r\), penalty \(\rho\), and
regularization \(\sigma\), the update is

$$
w^k = \Pi_{[\ell,u]}(v^k),
$$

$$
x^{k+1}
= (\Sigma + \sigma I + \rho A^\top A)^{-1}
  \left(\sigma x^k + r + \rho A^\top(2w^k - v^k)\right),
$$

$$
v^{k+1} = v^k + Ax^{k+1} - w^k.
$$

## Speedup

The recorded benchmark ran 200 ADMM iterations per solve.

| implementation | time per solve |
|---|---:|
| direct NumPy Python | `2017.354 us` |
| generated C after code generation | `14.831 us` |
| generated C after C optimization | `6.351 us` |

The generated C solver is `136.0x` faster than direct Python. The C optimization
pass adds another `2.34x` speedup over the initial generated C solver, for a
total speedup of `317.6x` from direct Python to optimized C. The optimized
solver was kept after verification with worst max-absolute error `2.220e-16`.

## Optimization Trick

The main trick for portfolio is combining several small structural
specializations. The optimized solver replaces the right-hand-side build with
direct scalar formulas, replaces sparse state-update helpers with direct scalar
assignments, replaces the projection helper with direct clamps, inlines the
fixed four-variable linear solve, updates state in place while preserving the
final scratch arrays, forces inlining of a hot helper, and adds a verified
fixed-active-set path after the startup iterations.

The largest wins came from inlining the solve, updating state in place, and
using the verified active-set specialization while keeping the generic path for
early iterations and other iteration counts.
