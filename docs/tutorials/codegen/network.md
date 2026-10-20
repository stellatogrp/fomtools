# Network Flow Code Generation Speedup

This tutorial benchmarks a small capacitated network-flow problem. It compares a
direct NumPy implementation with a verified generated C solver, then reports the
speedup from code generation and the subsequent C optimization pass.

Open the companion [marimo notebook](network.html), or edit the source locally
from the repository root:

```bash
uv run marimo edit docs/tutorials/codegen/network.py
```

## Problem

The problem is a capacitated min-cost network-flow linear program:

$$
\begin{array}{ll}
    \text{minimize} & c^T z \\
    \text{subject to} & A_s z \le b_s, \\
    & A_d z = x, \\
    & 0 \le z \le u.
\end{array}
$$

Here $z$ is the edge-flow vector, $c$ contains edge costs, $A_s$ maps flows
to supply-node usage, $b_s$ contains supply capacities, $A_d$ maps flows to
demand-node balance, $x$ is the demand parameter, and $u$ contains edge
capacities. The notebook uses a complete bipartite graph from supply nodes to
demand nodes.

## Method

The solver uses a primal-dual projected update. With step size $\eta$, supply
dual variable $v$, and demand-balance dual variable $w$, each iteration is

$$
z^{k+1}
= \Pi_{[0,u]}\left(
    z^k - \eta(c + A_s^T v^k - A_d^T w^k)
\right),
$$

$$
v^{k+1}
= \max\left\{
    v^k + \eta\left(-b_s + A_s(2z^{k+1} - z^k)\right),
    0
\right\},
$$

$$
w^{k+1}
= w^k + \eta\left(x - A_d(2z^{k+1} - z^k)\right).
$$

Both the NumPy reference and the generated C solver run the same fixed iteration
map.

## Speedup

The recorded benchmark ran 200 primal-dual projected iterations per solve.

| implementation | time per solve |
|---|---:|
| direct NumPy Python | `2358.812 us` |
| generated C after code generation | `10.613 us` |
| generated C after C optimization | `2.593 us` |

The generated C solver is `222.3x` faster than direct Python. The C
optimization pass adds another `4.09x` speedup over the initial generated C
solver, for a total speedup of `909.8x` from direct Python to optimized C. The
optimized solver was kept after verification with worst max-absolute error
`1.443e-14`.

## Optimization Trick

The main trick for network flow is scalarizing the whole 11-state recurrence.
The optimized solver replaces generic clipping and nonnegative-projection
helpers with direct scalar clamp and ReLU code, splits the final
residual-producing step from the non-final loop to remove a branch from most
iterations, and fuses final state and solution stores where that was neutral or
slightly faster.

In this example, fewer arithmetic operations were not always faster; register
pressure and compiler scheduling mattered more than some algebraic reductions.
