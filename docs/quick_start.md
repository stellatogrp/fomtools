# Quick start

This guide walks through how to use FOMTools: linear expressions, steps, proxable
functions, objectives, and the fixed-point encoding. It focuses on what you write as a
user, not on how the package solves the resulting problem underneath.

## The problem: verifying FISTA

Suppose you want to solve a LASSO-type problem

```
minimize_w  (1/2) ||Dw - x||_2^2 + lambda ||w||_1
```

where $D$ is a fixed data matrix and $x$ is a parameter that is only known to lie in some
uncertainty set, for instance sensor noise or a range of possible measurements. FISTA is a
standard algorithm for this problem: starting from an iterate $w_0$, it repeats a
gradient step on the smooth part followed by a soft-thresholding step for the $\ell_1$-norm, plus a
momentum term that mixes the current and previous iterate.

The question FOMTools answers is: **for every $x$ in the uncertainty set, how much does the
iterate still move after $K$ steps?** That is, what is the worst-case value, over all
admissible $x$, of a residual such as $\|s^{k+1} - s^k\|$? A small worst-case residual after
$K$ iterations is a certificate that FISTA has converged for every possible instance of the
parameter, not just the one you happened to test.

FOMTools lets you answer this by writing the FISTA iteration almost exactly as you would
write a concrete implementation, and then asking it to maximize the residual over the
declared parameter set. The rest of this guide builds up that code.

## [Setting up the problem](api/verifier.md)

You declare the problem parameter $x$, the initial iterate $s_0$, and a `Verifier` object
that will hold everything you build. Bounds passed to `add_param` describe the uncertainty
set for $x$; bounds passed to `add_iterate` describe where the algorithm starts.

```python
import fomtools as ft
import numpy as np

# problem parameters and setup
lambd, eta, K = 0.1, 0.01, 10
m, n = 10, 15
D = np.random.normal(size=(m, n))
VP = ft.Verifier(method="mip", solver="gurobi")

# initialize VP parameters + iterates
x = VP.add_param(m, lb=0, ub=1)
s = [None for _ in range(K + 1)]
s[0] = VP.add_iterate(n, lb=0, ub=0)
wk = s[0]
beta_k = 1
results = []

for k in range(K):
    # core algorithm logic
    grad = D.T @ (D @ wk - x)
    s[k + 1] = VP.step.soft_threshold(wk - eta * grad, lambd=lambd * eta)
    beta_kplus1 = 0.5 * (1 + np.sqrt(1 + 4 * beta_k ** 2))
    w_kplus1 = s[k + 1] + (beta_k - 1) / beta_kplus1 * (s[k + 1] - s[k])

    # solving the VP
    VP.set_objective_norm(s[k + 1] - s[k], np.inf)
    results.append(VP.solve())

    # setting up for next loop iteration
    wk, beta_k = w_kplus1, beta_kplus1
```

Each line of this loop mirrors the mathematical description of FISTA: the gradient step, the
soft-thresholding step, and the momentum update. $w_{k+1}$ does not need to be declared as a
new iterate; you just write the affine combination directly, the same way you would with
plain numpy arrays. After each iteration you set the objective to the residual you care about
and call `solve()` to get the worst-case value for that iteration.

## [Choosing a backend](api/verifier.md)

`method` is the only line you need to change to switch backends; the rest of the FISTA code
above is untouched. MIP encodes the verification problem exactly (via Gurobi or HiGHS) and
returns the true worst-case value, at the cost of solving a combinatorial program that scales
poorly as the horizon or dimensions grow. SDP relaxes the same problem into a semidefinite
program (via Clarabel) and returns an upper bound instead of the exact value, but it solves in
polynomial time, so it remains tractable on problems where MIP stalls.

```python
# exact worst case, combinatorial cost
VP = ft.Verifier(method="mip", solver="gurobi")
```

```python
# polynomial-time upper bound
VP = ft.Verifier(method="sdp", solver="clarabel")
```

The SDP relaxation can be tightened with optional cuts, passed as extra keyword arguments to
`Verifier`; each is ignored (with a warning) when `method` is not `"sdp"`.

```python
# a tighter, more expensive SDP relaxation
VP = ft.Verifier(
    method="sdp",
    solver="clarabel",
    rlt_cuts=True,
    triangle_cuts=True,
    full_psd_cone=True,
)
```

A third option, the simulator, gives up the formal worst-case guarantee for speed: it
evaluates the algorithm on a large sample of parameters and initial iterates and reports the
empirical worst case, together with a high-probability bound on the population worst case
from a KL-inverse statistical guarantee. It scales to much larger problems and horizons than
MIP or SDP, and is a good way to get a quick read on an algorithm before committing to a
formal solve.

```python
# empirical worst case with a statistical guarantee, on NumPy/SciPy or CuPy
VP = ft.Verifier(method="simulator", solver="numpy")
```

A common workflow is to run all three on a small instance and compare: if the MIP and SDP
values are close, the (much cheaper) SDP bound can be trusted on larger instances of the same
problem, and if the simulator's empirical worst case and risk bound track the MIP value
closely, the simulator can be trusted to scale the analysis further still.

| Backend | Result | Cost |
| --- | --- | --- |
| MIP | Exact worst case | Most expensive, practical for smaller horizons and dimensions |
| SDP | Upper bound | Cheaper than MIP, scales further |
| Simulator | Empirical worst case + statistical bound | Cheapest, scales to the largest problems |

## [Linear expressions](api/linearexpression.md)

Every quantity you write in FOMTools (parameters, iterates, and any combination of them)
behaves like a numpy array. You can add expressions, scale them, and left-multiply them by
numpy arrays or scipy sparse matrices, and the result is itself a valid expression you can
keep building on.

```python
# different ways to declare leaf objects
m, n = 10, 15
x = VP.add_param(m, lb=0, ub=1)
z0 = VP.add_iterate(n, lb=0, ub=0)

# arbitrary combinations
A = np.random.normal(size=(n, n))
B = np.random.normal(size=(n, m))
y = A @ (z0 - 2 * B @ x)

# arbitrary constraints
VP.add_constraint(np.ones((1, m)) @ x == 1)
```

Two kinds of objects need bounds when you create them: parameters (`add_param`), which
describe the uncertainty set, and iterates (`add_iterate`), which describe the algorithm's
starting point or the output of a step. Anything built out of those, like $y$ above, gets its
bounds automatically and does not need its own `lb`/`ub`. You can also impose arbitrary
polyhedral constraints on any expression with `==`, `<=`, and `>=`, which is useful for
encoding structure in the parameter set, such as a probability simplex.

## [Steps](api/steps.md)

A step is how you write one iteration of your algorithm. Affine operations (gradient
evaluations on quadratics, momentum updates, linear solves) are just ordinary expression
arithmetic, as in the FISTA example above. For nonlinear operations, FOMTools provides three
step functions:

```python
s_next = VP.step.relu(s - eta * grad)
s_next = VP.step.soft_threshold(s - eta * grad, lambd=lambd * eta)
s_next = VP.step.satlin(s - eta * grad, lb=l, ub=u)
```

- `relu` computes the elementwise ReLU, $\max(0, \cdot)$.
- `soft_threshold` computes the elementwise soft-thresholding operator with threshold
  `lambd`, the proximal operator of $\lambda\|\cdot\|_1$.
- `satlin` clips its input elementwise to `[lb, ub]`, the proximal operator of the indicator
  of a box.

## [Proxable functions](api/proxablefunctions.md)

Many first-order methods are built from proximal gradient steps on a convex function $f$,
for a step size $\gamma > 0$. FOMTools covers 24 such functions out of the box, so you rarely
need to hand-write a step. The $\ell_1$-norm, the indicator of the nonnegative orthant, and the
indicator of a box are common examples, but the library also covers convex quadratics and the
indicators of a polyhedron, a simplex, and an $\ell_1$-ball, among others.

A proxable function is applied with `prox(function, expr, gamma)`. For an indicator, the
equivalent `project(set, expr)` takes the set itself and is the more natural spelling, since
projection does not depend on `gamma`.

```python
from fomtools.proxable_functions import NormL1
from fomtools.sets import Box

# a proximal gradient step, two equivalent spellings
z_next = VP.prox(NormL1(lambd), z - eta * grad, eta)
z_next = VP.project(Box(l, u), z - eta * grad)
```

Both entry points return an expression, so a proximal gradient step is a single line of user
code.

## [Objectives](api/verifier.md#fomtools.Verifier.set_objective_norm)

The verification problem maximizes a norm of an arbitrary expression, so you are not limited
to the standard fixed-point residual $\|s^{k+1} - s^k\|$: you can use scaled residuals,
residuals involving the problem parameters, or distances to a reference expression. FOMTools
supports four objectives: the $\ell_1$-norm, the squared $\ell_2$-norm, the $\ell_\infty$-norm, and a zero objective
for feasibility checks. Set the objective with `set_objective_norm(expr, ord)`, where `expr`
is an expression and `ord` selects the norm. `expr` can also be a list of expressions, which
FOMTools stacks into a single vector before applying the norm, which is useful for composite metrics
such as the OSQP convergence check, which monitors both the primal and the dual residual.

```python
# OSQP convergence criterion
VP.set_objective_norm([M @ zk - wk, P @ zk + q + M.T @ yk], np.inf)
```

The dual residual involves the problem parameter $x$, which is fine, since the objective can
operate on any expression, including ones that mix iterates and parameters.

## [Fixed-point encoding](api/verifier.md#fomtools.Verifier.fixed_point)

Distance-to-optimality metrics require you to reference an optimal solution $z^\star$ of the
parametric problem. For first-order methods with contractive operators, $z^\star$ is a fixed point
of the algorithm operator $T$, that is, the vector satisfying $z^\star = T(z^\star, x)$, whose value
depends on the parameter $x$. You supply the operator as a Python function, together with
bounds on $z^\star$.

```python
# fixed point for ista
z_star = VP.fixed_point(
    n_dim=n,
    operator=lambda z: VP.step.soft_threshold(
        z - eta * D.T @ (D @ z - x), lambd=lambd * eta),
    lb=-100, ub=100,
)
VP.set_objective_norm(zk - z_star, 2)
```

`fixed_point` returns an expression for $z^\star$, which you can use in an objective just like any
other expression, as above, to maximize the distance between the $k$-th iterate and the fixed
point. The bounds you pass must hold for all parameters in the declared parameter set; they
can be conservative, but the tighter they are, the tighter the resulting verification bound
will be.
