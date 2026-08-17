# Nonnegative Least Squares Code Generation Speedup

This tutorial benchmarks projected gradient descent for a small nonnegative
least-squares problem. It compares a direct NumPy implementation with a verified
generated C solver, then reports the speedup from code generation and the
subsequent C optimization pass.

Open the companion [marimo notebook](nnls.html), or edit the source locally from
the repository root:

```bash
uv run marimo edit docs/tutorials/nnls.py
```

## Problem

The nonnegative least-squares instance is

$$
\begin{array}{ll}
    \text{minimize} & (1/2)\lVert Az - x \rVert_2^2 \\
    \text{subject to} & z \ge 0.
\end{array}
$$

The notebook uses a small dense matrix \(A\), a boxed input vector \(x\), and a
fixed positive starting vector.

## Method

The solver uses projected gradient descent:

$$
z^{k+1} = \max\{z^k - \eta A^\top(Az^k - x), 0\}.
$$

Both the NumPy reference and the generated C solver run the same update rule for
the same number of iterations.

## Speedup

The recorded benchmark ran 200 projected-gradient iterations per solve.

| implementation | time per solve |
|---|---:|
| direct NumPy Python | `690.375 us` |
| generated C after code generation | `9.352 us` |
| generated C after C optimization | `1.459 us` |

The generated C solver is `73.8x` faster than direct Python. The C optimization
pass adds another `6.41x` speedup over the initial generated C solver, for a
total speedup of `473.3x` from direct Python to optimized C. The optimized
solver was kept after verification with worst max-absolute error `1.388e-17`.

## Optimization Trick

The main trick for nonnegative least squares is removing generic helper
overhead from a tiny fixed update. The optimized solver precomputes the
parameter contribution to the affine update once per solve, inlines the fixed
three-by-three state multiply, replaces the identity state-update helper with a
direct assignment, and moves `solution.iter` and `solution.residual` writes out
of the per-iteration loop when only the final values are observable.

Several apparently useful memory-write reductions were tested but were not kept
because the compiler already handled them well or they hurt scheduling.
