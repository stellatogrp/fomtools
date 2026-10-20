FOMTools
========

[![PyPI](https://img.shields.io/pypi/v/fomtools.svg)](https://pypi.org/project/fomtools/)
[![Docs](https://img.shields.io/badge/docs-stellatogrp.github.io%2Ffomtools-blue.svg)](https://stellatogrp.github.io/fomtools/)

**Development, analysis, and code generation for first-order methods.**

[Documentation](https://stellatogrp.github.io/fomtools/) |
[PyPI](https://pypi.org/project/fomtools/) |
[Paper](https://arxiv.org/abs/2610.26743) |
[Benchmarks](https://github.com/stellatogrp/fomtools_experiments)

FOMTools is an open-source Python package for developing, analyzing, and deploying
first-order methods for parametric linear and convex quadratic optimization. You write the
iteration in Python, with syntax close to its mathematical description, and FOMTools records
it as a computation graph of affine and piecewise affine steps. From that single description
you can:

- **Verify** worst-case performance: a mixed-integer program maximizes a convergence residual
  over a set of problem instances and initial iterates (`method="mip"`).
- **Simulate** at scale: a batched CPU/GPU simulator collects trajectories and bounds the
  probability of exceeding a convergence tolerance (`method="simulator"`).
- **Generate code**: FOMTools builds the numerical data, factorizations, and matrix-vector
  routines, and an LLM coding agent writes and optimizes C, OpenMP, or CUDA code. Each phase
  is checked against a reference run on sampled inputs.
- **Solve CVXPY problems**: FOMTools works as a [CVXPYgen](https://github.com/cvxgrp/cvxpygen)
  backend, generating first-order solvers for parametric CVXPY problems.

## Example

The following code verifies FISTA on a family of Lasso problems
```
minimize        0.5 ||D s - x||_2^2 + lambda ||s||_1
```

where `s in R^n` is the optimization variable and the data vector `x` ranges over the box
`[0, 1]^m`. After each iteration it computes the worst-case fixed-point residual
`||s^{k+1} - s^k||_inf` over all such `x`.

```python3
import numpy as np
import fomtools as ft

# Problem data
m, n, lambd, K = 10, 15, 0.1, 10
D = np.random.normal(size=(m, n))
eta = 1 / np.linalg.norm(D, 2) ** 2

VP = ft.Verifier()  # MIP verification with HiGHS; use solver="gurobi" for Gurobi

# Parameter set and initial iterate
x = VP.add_param(m, lb=0, ub=1)
s = [VP.add_iterate(n, lb=0, ub=0)]
w, beta = s[0], 1.0

results = []
for k in range(K):
    # FISTA iteration
    s.append(VP.step.soft_threshold(w - eta * D.T @ (D @ w - x), lambd=lambd * eta))
    beta_old, beta = beta, 0.5 * (1 + np.sqrt(1 + 4 * beta**2))
    w = s[k + 1] + (beta_old - 1) / beta * (s[k + 1] - s[k])

    # Worst-case residual after k + 1 iterations
    VP.set_objective_norm(s[k + 1] - s[k], np.inf)
    results.append(VP.solve())
```

Steps can also be written with proximal operators and projections, e.g.,
`VP.prox(NormL1(lambd), v, eta)` or `VP.project(Box(l, u), v)`. See the
[documentation](https://stellatogrp.github.io/fomtools/) for the full list of proxable functions and sets.

## Beyond worst-case verification

**Simulation.** Write the same loop with `ft.Verifier(method="simulator")`, and give
parameters sample arrays of shape `(N, d)` (or let them be sampled from their bounds). The
simulator runs all samples as batched operations, on a GPU through CuPy when one is
available. It then reports summary statistics and a KL-inverse bound on the probability that
the residual exceeds a tolerance `epsilon`:

```python3
VP.solve(epsilon=1e-3)
VP["quantiles"][0.95], VP["risk_bound"], VP["throughput"]
```

**Code generation.** Pass the recorded iterates, unchanged, to `codegen`. It produces a
specialized solver together with a Python wrapper:

```python3
result = VP.codegen("out/fista", iterates=s, code_optimization=True)
```

Targets include embedded C, multicore CPUs (OpenMP), and GPUs (CUDA). The package works with
any coding agent.

**CVXPYgen.** You can model a parametric problem in CVXPY and generate both its
canonicalization code and a first-order solver for it:

```python3
import fomtools.cvxpygen as fcpg

fcpg.generate_solver(problem, "out/qp", fom="admm", param_bounds={q: (q_lb, q_ub)})
```

## Installation

FOMTools requires Python 3.12 or later. Install it from PyPI:

```bash
pip install fomtools
```

To install from source instead:

```bash
git clone https://github.com/stellatogrp/fomtools.git
cd fomtools
pip install -e .
```

Optional extras:

| Extra      | Description                                        |
|------------|----------------------------------------------------|
| `gurobi`   | Gurobi solver backend                              |
| `gpu`      | CuPy for GPU-accelerated simulation                |
| `cuda`     | cuDSS for direct linear solves in CUDA codegen     |
| `codegen`  | LLM coding-agent backends for code generation      |
| `cvxpygen` | CVXPYgen integration                               |
| `nn`       | Neural-network verification (PyTorch, JAX, Flax)   |
| `all`      | All of the above (GPU packages skipped on macOS)   |
| `dev`      | Development tools                                  |
| `docs`     | Documentation build dependencies                   |

For example, `pip install "fomtools[gurobi,codegen]"`, or
`pip install -e ".[gurobi,codegen]"` from source. Use `pip install "fomtools[all]"`
to install every optional feature.

### Agent skill

FOMTools ships an [Agent Skill](https://agentskills.io/) in
[`skills/fomtools`](skills/fomtools/SKILL.md) that teaches coding agents (Claude
Code, Codex, Cursor, and others) the FOMTools API. Install it with the
[`skills` CLI](https://github.com/vercel-labs/skills) (requires Node.js):

```bash
npx skills add stellatogrp/fomtools -g
```

See [Get started](https://stellatogrp.github.io/fomtools/get_started/) for details.

FOMTools is not a solver. It relies on mathematical programming solvers such as
[HiGHS](https://highs.dev) (the default) and [Gurobi](https://www.gurobi.com) to solve
the verification problems it builds.

## Examples

The [tutorials](https://stellatogrp.github.io/fomtools/tutorials/) cover sparse coding (Lasso), nonnegative least squares,
network optimization, optimal control, and portfolio rebalancing.

## Benchmarks

The benchmarks and the code to reproduce the experiments in our
[paper](https://arxiv.org/abs/2610.26743) (verification, simulation, and code generation on
CPU and GPU) are in a separate repository,
[stellatogrp/fomtools_experiments](https://github.com/stellatogrp/fomtools_experiments).

## Contributing

If you'd like to add a new example to our library, or implement a new feature,
please get in touch with us first to make sure that your priorities align with
ours.

## Team

FOMTools is developed by Vinit Ranjan, Daniel Deza, Anushka Acharya, Jisun Park, and
Bartolomeo Stellato.

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
