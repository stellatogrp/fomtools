# ![FOMTools](assets/logo/logo.svg#only-light){ .fomtools-logo } ![FOMTools](assets/logo/logo-dark.svg#only-dark){ .fomtools-logo }

FOMTools is a Python package that provides tools to verify the performance
of first-order iterative methods applied on parametric quadratic optimization.

To install the package and the FOMTools agent skill for coding agents, see
[Get started](get_started.md).

## Quick Start

This example uses the default open-source HiGHS solver to verify the behavior of projected
gradient descent applied to the parametric quadratic program

$$
\begin{array}{ll}
	\text{minimize} & \frac{1}{2} z^T P z + q^T z \\
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

This short script is a basic example of what FOMTools can do: verifying
worst-case behavior of an iterative algorithm under parameter uncertainty using
mixed-integer optimization.

Beyond this example, FOMTools supports verification of a wide range of iterative methods,
including projected gradient methods, proximal operators, and algorithms with nonlinearities.
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
