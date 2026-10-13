# FOMTools

FOMTools is a Python package that provides tools to verify the performance
of first-order methods in parametric quadratic optimization.

## Quick Start

This example verifies the behavior of ReLU gradient iteration
with uncertain parameters using the default open-source HiGHS solver.

```python
import numpy as np
from fomtools.verifier import Verifier


K = 5          # number of iterations
n = 3           # dimension of the iterate
step_size = 0.01
q_offset = 1.0

rng = np.random.default_rng(0)
vp = Verifier(params={"output_flag": False})

z = [None] * (K + 1)
# set up initial iterate
z[0] = vp.add_iterate(ndim=n, lb=-1.0, ub=-1.0)

q_nominal = rng.normal(size=n)
P = rng.normal(size=(n, n))
P = P @ P.T

# q is a problem parameter
q_param = vp.add_param(ndim=n, lb=q_nominal - q_offset, ub=q_nominal + q_offset)

# verification loop
results = []

for k in range(1, K + 1):
    z[k] = vp.step.relu(z[k - 1] - step_size * (P @ z[k - 1] + q_param))

    # objective: maximize ||z_k - z_{k-1}||_∞
    vp.set_objective_norm(z[k] - z[k - 1], ord=np.inf)

    # solve the resulting verification problem
    res = vp.solve(convex_hull_cuts=True)
    results.append(res)

    # Sum all certified successive residual bounds from z[0] to z[k].
    vp.post_process(z[k], z[0], results)

print(f"opt q_param at last K: {vp.get_solution_value(q_param)}")
print(f"Worst-case residual upper bounds: {results}")

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

## Returned bounds and numerical checks

`solve()` returns a finite MIP best bound, including after a time limit when one
is available. It raises `RuntimeError` for unusable termination states or missing
bounds. An incumbent is a feasible example and is available separately through
`vp.canonicalizer._solved_copy["objVal"]`; its native status is available through
`["status"]`. `get_solution_value` requires an incumbent.

SDP solves require an accurate optimal termination and return the dual objective.
Inaccurate, infeasible, unbounded, and early terminations raise `RuntimeError`.
The solved copy retains `last_solver.solution` for native Clarabel and
`last_problem` for CVXPY. These are numerical bounds subject to solver tolerances,
not exact-arithmetic proofs. OBBT also uses numerical LP solves; outward padding
reduces sensitivity to small solver errors but is not a rigorous certificate.

The simulator returns sampled residuals. Its default infinity-norm differences
follow public step/prox outputs, excluding internal KKT and prox variables.
For algorithms with several state blocks or intermediate operations, set an
explicit objective at each iteration. Statistical risk bounds require independent
evaluation samples from the specified distribution and a fixed tolerance.

`post_process` is valid only when its list contains upper bounds on **every**
successive infinity-norm difference from the reference to the target over the same
family. Do not pass a final-step bound alone, an incumbent, or a sample maximum.
Generated-code checks likewise compare sampled numerical executions and do not
prove floating-point equivalence for all inputs.
