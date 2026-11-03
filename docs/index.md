# ![FOMTools](assets/logo/logo.svg#only-light){ .fomtools-logo } ![FOMTools](assets/logo/logo-dark.svg#only-dark){ .fomtools-logo }

FOMTools is a Python package for developing, analyzing, and deploying first-order methods
for parametric linear and convex quadratic optimization. You write the iteration once in
Python, and FOMTools uses that description to

- **verify** worst-case performance with mixed-integer optimization;
- **simulate** it at scale on CPU or GPU, with probabilistic bounds;
- **generate code**: an LLM coding agent writes and optimizes a C, OpenMP, or CUDA solver,
  checked against a reference run at every step;
- **solve CVXPY problems** as a [CVXPYgen](https://github.com/cvxgrp/cvxpygen) backend.

[![A coding agent writes a lasso script with the FOMTools skill; FOMTools verifies the worst case, then an agent translates and optimizes an embedded C solver](assets/codegen-lasso.gif)](assets/codegen-lasso.gif)

A real run, time-lapsed (tap to enlarge). Claude Code, with the FOMTools skill, wrote the lasso script.
FOMTools computes the worst-case residual, then a coding agent translates the iteration to
C and makes it 2.25x faster in two rounds, over 2,000x faster than NumPy. Every
rewrite is checked against the reference. See the
[code generation tutorials](tutorials/codegen.md) for more examples.

To install the package and the FOMTools agent skill for coding agents, see
[Get started](get_started.md).

## Quick Start

This example uses the default open-source HiGHS solver to verify the behavior of projected
gradient descent applied to the parametric quadratic program

$$
\begin{array}{ll}
	\text{minimize} & (1/2) z^T P z + q^T z \\
	\text{subject to} & z \geq 0,
\end{array}
$$

where $z \in \mathbf{R}^n$ is the optimization variable, $P \in \mathbf{R}^{n \times n}$ is a fixed symmetric positive semidefinite matrix, and $q \in [-1, 1]^n$ is a problem parameter.
Starting from $z^0 = 0$, projected gradient descent with step size $\eta$ (`eta` in the code) uses the update

$$
z^{k+1} = \Pi_{\mathbf{R}^n_+}\left(z^k - \eta(Pz^k + q)\right), \quad k = 0, 1, \dots,
$$

where the projection onto the nonnegative orthant $\Pi_{\mathbf{R}^n_+}$ is the `relu` step.
After $K$ iterations, FOMTools computes the worst-case fixed-point residual $\|z^K - z^{K-1}\|_\infty$ over all $q \in [-1, 1]^n$.

```python
import numpy as np
import fomtools as ft

K, n, eta = 5, 3, 0.01
A = np.random.default_rng(0).normal(size=(n, n))
P = A @ A.T

vp = ft.Verifier()                      # HiGHS by default
q = vp.parameter(n, lb=-1, ub=1)        # q in [-1, 1]^n
z = vp.iterate(n, lb=0, ub=0)           # z^0 = 0

for k in range(K):
    z_prev, z = z, vp.relu(z - eta * (P @ z + q))

vp.maximize(ft.norm(z - z_prev, np.inf))
print("worst-case residual:", vp.solve())
print("worst-case q:", q.value)
```

`solve()` returns an upper bound on the worst case: the exact value, up to the MIP gap.
`q.value` is the parameter that attains it.
The `Verifier` records the whole iteration, so the loop only keeps the last two iterates for the objective.
Pass `verbose=True` to `Verifier` or to `solve` to see the solver logs.

To generate a solver, keep all the iterates in a list and pass them to `codegen`.
It produces a specialized C solver with a Python wrapper.
A coding-agent CLI (`claude` by default) writes the solver, so it must be installed and
logged in:

```python
vp = ft.Verifier()
q = vp.parameter(n, lb=-1, ub=1)
z = [vp.iterate(n, lb=0, ub=0)]
for k in range(K):
    z.append(vp.relu(z[-1] - eta * (P @ z[-1] + q)))

result = vp.codegen("out/pgd", iterates=z, code_optimization=True)
```

For a guided introduction, see the [FOMTools tutorials](tutorials/index.md), which cover
[verification](tutorials/verification.md) and [code generation](tutorials/codegen.md)
examples from optimization and control.

FOMTools relies on mathematical programming solvers such as HiGHS (the default) and Gurobi to solve the
resulting verification problems.

## Benchmarks

The benchmarks and the code to reproduce the experiments in our
[paper](https://arxiv.org/abs/2610.26743) are in a separate repository,
[stellatogrp/fomtools_experiments](https://github.com/stellatogrp/fomtools_experiments).

## Development

FOMTools is developed by Vinit Ranjan, Daniel Deza, Anushka Acharya, Jisun Park, and
Bartolomeo Stellato. The mixed-integer verification builds on our earlier paper
[Exact Verification of First-Order Methods via Mixed-Integer Linear Programming](https://arxiv.org/abs/2412.11330)
with Stefano Gualandi and Andrea Lodi.

- Source code: [github.com/stellatogrp/fomtools](https://github.com/stellatogrp/fomtools)
- PyPI: [pypi.org/project/fomtools](https://pypi.org/project/fomtools/)
- Installation: see [Get started](get_started.md)

## Citing

If you use FOMTools for published work, we encourage you to cite our
[paper](https://arxiv.org/abs/2610.26743):

```bibtex
@article{ranjan2026fomtools,
  title   = {{FOMTools}: Development and {AI}-Assisted Code Generation for First-Order Methods},
  author  = {Ranjan, Vinit and Deza, Daniel and Acharya, Anushka and Park, Jisun and Stellato, Bartolomeo},
  journal = {arXiv preprint arXiv:2610.26743},
  year    = {2026},
}
```

The mixed-integer verification method is described in
[Exact Verification of First-Order Methods via Mixed-Integer Linear Programming](https://arxiv.org/abs/2412.11330).
