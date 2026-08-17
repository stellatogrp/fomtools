# Optimal Control Code Generation Speedup

This tutorial benchmarks a small finite-horizon control problem for a linear
dynamical system. It compares a direct NumPy implementation with a verified
generated C solver, then reports the speedup from code generation and the
subsequent C optimization pass.

Open the companion [marimo notebook](control.html), or edit the source locally
from the repository root:

```bash
uv run marimo edit docs/tutorials/control.py
```

## Problem

The problem is a short-horizon linear-quadratic control problem:

$$
\begin{array}{ll}
    \text{minimize} &
        (1/2)\sum_{t=0}^{H-1} x_t^\top Q x_t
        + (1/2)\sum_{t=0}^{H-2} u_t^\top R u_t \\
    \text{subject to} &
        x_{t+1} = A x_t + B u_t,\quad t=0,\ldots,H-2, \\
    &   x_0 = x_{\mathrm{init}}, \\
    &   u_{\min} \le u_t \le u_{\max},\quad t=0,\ldots,H-2.
\end{array}
$$

The notebook uses \(H=2\), condenses the dynamics into a quadratic program in a
single decision vector \(z\), and encodes the fixed initial state and input box
as bounds \(\ell \le z \le u\):

$$
\begin{array}{ll}
    \text{minimize} & (1/2)z^\top Pz + q^\top z \\
    \text{subject to} & \ell \le z \le u.
\end{array}
$$

## Method

The solver uses the alternating direction method of multipliers. With \(M=I\),
penalty \(\rho\), regularization \(\sigma\), and projection
\(\Pi_{[\ell,u]}\), each iteration is

$$
w^k = \Pi_{[\ell,u]}(v^k),
$$

$$
z^{k+1}
= (P + \sigma I + \rho I)^{-1}
  \left(\sigma z^k - q + \rho(2w^k - v^k)\right),
$$

$$
v^{k+1} = v^k + z^{k+1} - w^k.
$$

Both the NumPy reference and the generated C solver run the same update rule for
the same number of iterations.

## Speedup

The recorded benchmark ran 200 ADMM iterations per solve.

| implementation | time per solve |
|---|---:|
| direct NumPy Python | `2817.667 us` |
| generated C after code generation | `153.596 us` |
| generated C after C optimization | `29.704 us` |

The generated C solver is `18.3x` faster than direct Python. The C optimization
pass adds another `5.17x` speedup over the initial generated C solver, for a
total speedup of `94.9x` from direct Python to optimized C. The optimized solver
was kept after verification with worst max-absolute error `5.031e-16`.

## Optimization Trick

The main trick for optimal control is specializing the fixed linear solve and
state update. The optimized solver inlines the fixed-size solve instead of
calling the generic solve path, removes avoidable branch and indexing overhead,
uses direct assignments for the known state-update structure, applies direct
bound logic for the projection step, folds known constants into the solve
right-hand side, and moves final-only state and residual bookkeeping out of the
hot loop where that preserves the reported result.

The largest win came from replacing generic linear algebra and state-update
helpers with code specialized to this fixed problem structure.
