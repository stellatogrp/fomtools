# User guide

This guide walks through the core components of FOMTools: linear expressions, steps,
proxable functions, objectives, and the fixed-point encoding.

## [Architecture overview](api/index.md)

The user writes the first-order method as a sequence of operations on expression objects
that represent iterates and parameters, and that the package maintains internally as affine
functions of a set of leaf variables. Each piecewise affine (PWA) step creates a new leaf
and records the step. Once the user sets an objective and calls `solve`, the package
decomposes the steps into the primitives that the chosen backend supports, canonicalizes
the verification problem into MIP, SDP, or simulator form, and dispatches to the underlying
solver. The same specification also drives a code generation path, which emits standalone
solver code for embedded and parallel targets.

## [Workflow walkthrough](api/verifier.md)

We illustrate the user experience on the FISTA example. We import the package, set up the
problem data, create a `Verifier` object, declare the problem parameter `x` and the initial
iterate `s[0]`, and then write the FISTA iterations in a loop. The code mirrors the
mathematical description of FISTA, with the soft-thresholding step, the momentum update, and
the objective each on a single line.

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

The momentum variable `w_kplus1` is not a new leaf. The package represents it internally as
an affine combination of the leaf iterates `s[k+1]` and `s[k]`, which is the automatic
substitution that eliminates the auxiliary momentum sequence without any manual algebra. The
code reads almost identically to a concrete implementation of FISTA.

## [Linear expressions](api/linearexpression.md)

Internally, FOMTools represents every quantity in the verification problem as an affine
combination `Ay + b` of a stacked vector `y` of leaf variables. A leaf is an expression that
cannot be written as an affine combination of others: parameters, initial iterates, and
outputs of PWA steps are always leaves. Through operator overloading, expression objects
support addition, scalar multiplication, and left multiplication by numpy arrays or scipy
sparse matrices, so the package tracks arbitrary affine combinations symbolically.

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

The expression `y` above is not a new leaf: its internal representation records the affine
map from the leaves `z0` and `x`. This mechanism is what enables automatic dimension
reduction, since explicit affine steps such as gradient computations and momentum updates
never introduce new variables into the verification problem. Interval arithmetic propagates
elementwise bounds on each expression through affine operations. Arbitrary polyhedral
constraints can also be imposed on any expression through the `==`, `<=`, and `>=`
operators, as above, which is useful for encoding structure in the parameter set such as a
probability simplex.

## [Steps](api/steps.md)

A step is a primitive operator that maps an affine input to a possibly PWA output, and is
the unit at which the package interfaces with each backend. Affine steps, such as gradient
evaluations on quadratics, momentum updates, and linear solves, live inline in the affine
expression algebra and introduce no new leaves. The three PWA steps, ReLU, soft-thresholding,
and saturated linear, each create a new leaf and record the activation pattern that the
canonicalizer needs:

| Step | MIP | SDP | Simulator |
| --- | --- | --- | --- |
| Affine | native | native | native |
| ReLU | native | native | native |
| Soft-thresholding | native | decomposed | native |
| Saturated linear | native | decomposed | native |

The MIP and simulator backends handle all three natively, while the SDP backend natively
supports only ReLU; a step-decomposition layer rewrites soft-thresholding and saturated
linear into ReLU primitives automatically, so the same user code runs unchanged on every
backend.

```python
s_next = VP.step.relu(s - eta * grad)
s_next = VP.step.soft_threshold(s - eta * grad, lambd=lambd * eta)
s_next = VP.step.satlin(s - eta * grad, lb=l, ub=u)
```

## [Proxable functions](api/proxablefunctions.md)

A proxable function in FOMTools is a convex function `f` whose proximal operator the
package can encode exactly, for a step size `gamma > 0`. The library covers 24 functions.
Most have a closed form in the PWA primitives above, so they compile to one step plus affine
arithmetic: the ℓ1-norm becomes a single soft-thresholding step, the indicator of the
nonnegative orthant a single ReLU, and the indicator of a box a single saturated linear step.
The rest, such as convex quadratics and the indicators of a polyhedron, a simplex, or an
ℓ1-ball, have no such closed form; for these the package encodes the optimality conditions
of the proximal operator directly, as linear equalities together with a complementarity
condition on the multipliers.

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
code and the decomposition into steps stays hidden.

## [Objectives](api/verifier.md#fomtools.Verifier.set_objective_norm)

The verification problem maximizes a norm of an arbitrary expression, so custom performance
metrics can go beyond the standard fixed-point residual `‖s^{k+1} - s^k‖`, including scaled
residuals, residuals involving the problem parameters, and distances to a reference
expression. The package supports four objectives: the ℓ1-norm, the squared ℓ2-norm, the
ℓ∞-norm, and a zero objective for feasibility checks. The objective is set with
`set_objective_norm(expr, ord)`, where `expr` is an expression and `ord` selects the norm.
The objective also accepts a list of expressions, which the package stacks into a single
vector before applying the norm — useful for composite metrics such as the OSQP convergence
check, which monitors both the primal and the dual residual.

```python
# OSQP convergence criterion
VP.set_objective_norm([M @ zk - wk, P @ zk + q + M.T @ yk], np.inf)
```

The list interface stacks both residual vectors and maximizes the ℓ∞-norm of the result. The
dual residual involves the problem parameter `x`, which is permitted because the objective
operates on arbitrary expressions.

## [Fixed-point encoding](api/verifier.md#fomtools.Verifier.fixed_point)

Distance-to-optimality metrics require the verification problem to encode an optimal
solution `z*` of the parametric problem. For first-order methods with contractive operators,
`z*` is a fixed point of the algorithm operator `T`, that is, the vector satisfying
`z* = T(z*, x)`, whose value depends on the parameter `x`. The operator is supplied as a
Python function, together with bounds on `z*`.

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

Internally, `fixed_point` creates a new iterate `z*` with the given bounds, applies the
operator to obtain `T(z*, x)`, and adds the equality constraint `z* = T(z*, x)`. The returned
expression then enters the objective, as above, which maximizes the distance between the
`k`-th iterate and the fixed point. The bounds on `z*` must hold for all parameters in the
declared parameter set; they can be overestimated, but the tightness of both the MIP
formulation and the SDP relaxation depends on reasonable estimates.

## [Implementation notes](api/canonicalizer.md)

The package separates the user-facing front end — expressions, steps, objectives, and the
fixed-point encoding — from the back-end canonicalizers and solvers. All canonicalizers
implement a common interface: accept variables with bounds, add constraints, set an
objective norm, solve, and retrieve solution values. Adding a new backend therefore amounts
to implementing this interface, without touching the upstream pipeline. After a solve, the
optimal value of any expression can be queried, including non-leaf expressions, which the
package reconstructs from the leaf solution through the stored affine representation. All
coefficient matrices are stored in sparse format using `scipy.sparse`, and interval-arithmetic
bound propagation through affine maps is lazy, so bounds are recomputed only when a
canonicalizer consumes them.
