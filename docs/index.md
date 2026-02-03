# Welcome to FOM Tools's documentation!

FOMTools is a Python package that provides tools to verify the performance
of first-order methods in parametric quadratic optimization.

## Quick Start

This example verifies the behavior of ReLU gradient iteration
with uncertain parameters using FOMTools and Gurobi.

```python
import numpy as np
from fomtools.verifier import Verifier


K = 10          # number of iterations
n = 3           # dimension of the iterate
step_size = 0.01
q_offset = 1.0

vp = Verifier(solver="gurobi") # initialize verifier

z = [None] * (K + 1)
# set up initial iterate
z[0] = vp.add_iterate(dim=n, lb=-1.0, ub=-1.0)

q_nominal = np.random.normal(size=n)
P = np.random.normal(size=(n, n))
P = P @ P.T

# q is a problem parameter
q_param = vp.add_param(dim=n, lb=q_nominal - q_offset, ub=q_nominal + q_offset)

# verification loop
results = []

for k in range(1, K + 1):
    z[k] = vp.step.relu(z[k - 1] - step_size * (P @ z[k - 1] + q_param))

    # objective: maximize ||z_k - z_{k-1}||_∞
    vp.set_infinity_norm_objective(z[k] - z[k - 1])

    # solve the resulting verification problem
    res = vp.solve(huchette_cuts=True)
    results.append(res)

    # post-process results (e.g., tightening bounds, logging)
    vp.post_process(z[k], z[0], results)

print(f"opt q_param at last K: {vp.get_solution_value(q_param)}")
print(f"vp_gurobi residuals: {results}")

```

This short script is a basic example of what FOMTools can do. It demonstrates how to verify
worst-case behavior of an iterative algorithm under parameter uncertainty using
mixed-integer optimization.

Beyond this example, FOMTools supports verification of a wide range of iterative methods,
including projected gradient methods, proximal operators, and algorithms with nonlinearities.
For a guided introduction, see the FOMTools tutorial and example gallery,
which cover applications in optimization.

FOMTools relies on mathematical programming solvers such as Gurobi and Highs to solve the
resulting verification problems.

## Development

This repository is by Vinit Ranjan, Jisun Park, Stefano Gualandi, Andrea Lodi, Bartolomeo Stellato,
Daniel Deza, and Anushka Acharya and contains the Python source code to reproduce experiments
in our paper Exact Verification of First-Order Methods via Mixed-Integer Linear Programming.

Source Code: [https://github.com/stellatogrp/fomtools](https://github.com/stellatogrp/fomtools)
