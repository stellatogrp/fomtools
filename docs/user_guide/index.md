# User guide

This user guide is a more comprehensive companion to the [Quick Start](../quick_start.md),
covering more of FOMTools' capabilities.

## [Architecture overview](../api/index.md)

The user writes the first order method as a sequence of operations on expression objects
that represent iterates and parameters, and that the package maintains internally as affine
functions of a set of leaf variables. Each piecewise affine (PWA) step creates a new leaf
and records the step. Once the user sets an objective and calls `solve`, the package
decomposes the steps into the primitives that the chosen backend supports, canonicalizes
the verification problem into MIP, SDP, or simulator form, and dispatches to the underlying
solver. The same specification also drives a code generation path, which emits standalone
solver code for embedded and parallel targets.

## [Linear expressions](../api/linearexpression.md)

Internally, FOMTools represents every quantity in the verification problem as an affine
combination $Ay + b$ of a stacked vector $y$ of leaf variables. A leaf is an expression that
cannot be written as an affine combination of others: parameters, initial iterates, and
outputs of PWA steps are always leaves. Through operator overloading, expression objects
support addition, scalar multiplication, and left multiplication by numpy arrays or scipy
sparse matrices, so the package tracks arbitrary affine combinations symbolically.

```python
# reuse the imports and dimensions from the workflow above
demo = ft.Verifier(method="mip", solver="gurobi")
x_demo = demo.add_param(m, lb=0, ub=1)
z0 = demo.add_iterate(n, lb=0, ub=0)

# arbitrary combinations
A = np.random.normal(size=(n, n))
B = np.random.normal(size=(n, m))
y = A @ (z0 - 2 * B @ x_demo)

# arbitrary constraints
demo.add_constraint(np.ones((1, m)) @ x_demo == 1)
```

The expression `y` above is not a new leaf: its internal representation records the affine
map from the leaves `z0` and `x_demo`. This mechanism is what enables automatic dimension
reduction, since explicit affine steps such as gradient computations and momentum updates
never introduce new variables into the verification problem. Interval arithmetic propagates
elementwise bounds on each expression through affine operations. Arbitrary polyhedral
constraints can also be imposed on any expression through the `==`, `<=`, and `>=`
operators, as above, which is useful for encoding structure in the parameter set such as a
probability simplex.

## [Choosing a backend](../api/verifier.md)

FOMTools supports three backends for analyzing first order methods. All use the same
algorithm specification, but trade off exactness, cost, and scale.

| Backend | Result | Best fit | Main limitation |
| --- | --- | --- | --- |
| MIP-VP | Exact worst-case value | A tight deterministic certificate is required and the finite VP is tractable | Branch-and-bound can grow rapidly with the number of ambiguous PWA regions and iterations |
| SDP-VP ([experimental](sdp.md)) | Upper bound from a convex relaxation | A convex certificate is sufficient | The relaxation may be conservative, and the lifted PSD matrix can become expensive at larger dimensions and horizons |
| Simulator | Empirical worst case over samples with a statistical risk bound | Large-scale exploration, debugging, and data-driven assessment | Does not certify the worst case over every admissible instance |

All three backends support the same objective norms and share one user level algorithm
description. MIP and simulation handle ReLU, soft thresholding, and saturated linear steps
directly. The SDP backend represents the latter two through equivalent ReLU decompositions.

```python
# MIP: exact worst case analysis
demo = ft.Verifier(method="mip", solver="gurobi")
```

```python
# SDP (experimental): upper bound from a semidefinite relaxation
demo = ft.Verifier(method="sdp", solver="clarabel")
```

```python
# Simulator: empirical analysis at larger scale
demo = ft.Verifier(method="simulator", solver="numpy")
```

The SDP backend is experimental; its options, such as cuts that tighten the relaxation,
and its limitations are described in [Semidefinite relaxation (experimental)](sdp.md).

## [Steps](../api/steps.md)

A step is a primitive operator that maps an affine input to a possibly PWA output, and is
the unit at which the package interfaces with each backend. Affine steps, such as gradient
evaluations on quadratics, momentum updates, and linear solves, live inline in the affine
expression algebra and introduce no new leaves. The three PWA steps, ReLU, soft thresholding,
and saturated linear, each create a new leaf and record the activation pattern that the
canonicalizer needs.

```python
grad_demo = D.T @ (D @ z0 - x_demo)
l, u = -np.ones(n), np.ones(n)
s_relu = demo.step.relu(z0 - eta * grad_demo)
s_soft = demo.step.soft_threshold(z0 - eta * grad_demo, lambd=lambd * eta)
s_box = demo.step.satlin(z0 - eta * grad_demo, l=l, u=u)
```

## [Proxable functions](../api/proxablefunctions.md)

A proxable function in FOMTools is a convex function $f$ whose proximal operator the
package can encode exactly, for a step size $\gamma > 0$. The library covers 24 functions.
Most have a closed form in the PWA primitives above, so they compile to one step plus affine
arithmetic: the $\ell_1$ norm becomes a single soft thresholding step, the indicator of the
nonnegative orthant a single ReLU, and the indicator of a box a single saturated linear step.
The rest, such as convex quadratics and the indicators of a polyhedron, a simplex, or an
$\ell_1$ ball, have no such closed form. For these the package encodes the optimality conditions
of the proximal operator directly, as linear equalities together with a complementarity
condition on the multipliers.

A proxable function is applied with `prox(function, expr, gamma)`. For an indicator, the
equivalent `project(set, expr)` takes the set itself and is the more natural spelling, since
projection does not depend on $\gamma$.

```python
from fomtools.proxable_functions import NormL1
from fomtools.sets import Box

# equivalent to s_soft and s_box above, respectively
z_soft = demo.prox(NormL1(lambd), z0 - eta * grad_demo, eta)
z_box = demo.project(Box(l, u), z0 - eta * grad_demo)
```

Both entry points return an expression, so a proximal gradient step is a single line of user
code and the decomposition into steps stays hidden.

## [Objectives](../api/verifier.md#fomtools.Verifier.set_objective_norm)

The verification problem maximizes a norm of an arbitrary expression, so custom performance
metrics can go beyond the standard fixed point residual $\lVert s^{k+1} - s^k \rVert$, including scaled
residuals, residuals involving the problem parameters, and distances to a reference
expression. The package supports four objectives: the $\ell_1$ norm, the squared $\ell_2$ norm, the
$\ell_\infty$ norm, and a zero objective for feasibility checks. The objective is set with
`set_objective_norm(expr, ord)`, where `expr` is an expression and `ord` selects the norm.
The objective also accepts a list of expressions, which the package stacks into a single
vector before applying the norm, useful for composite metrics such as the OSQP convergence
check, which monitors both the primal and the dual residual.

```python
# illustrative OSQP residuals with bounded primal and dual variables
M, P = np.eye(n), D.T @ D
q = -D.T @ x_demo
zk = demo.add_iterate(n, lb=-1, ub=1)
wk = demo.add_iterate(n, lb=-1, ub=1)
yk = demo.add_iterate(n, lb=-1, ub=1)
demo.set_objective_norm([M @ zk - wk, P @ zk + q + M.T @ yk], np.inf)
```

The list interface stacks both residual vectors and maximizes the $\ell_\infty$ norm of the result. The
dual residual involves the problem parameter `x_demo` through `q`, which is permitted because the objective
operates on arbitrary expressions.

## [Fixed point encoding](../api/verifier.md#fomtools.Verifier.fixed_point)

Distance to optimality metrics require the verification problem to encode an optimal
solution $z^\star$ of the parametric problem. For first order methods with contractive operators,
$z^\star$ is a fixed point of the algorithm operator $T$, that is, the vector satisfying
$z^\star = T(z^\star, x)$, whose value depends on the parameter $x$. The operator is supplied as a
Python function, together with bounds on $z^\star$.

```python
# fixed point for ista
zk = z0
for _ in range(K):
    zk = demo.step.soft_threshold(
        zk - eta * D.T @ (D @ zk - x_demo), lambd=lambd * eta)
z_star = demo.fixed_point(
    n_dim=n,
    operator=lambda z: demo.step.soft_threshold(
        z - eta * D.T @ (D @ z - x_demo), lambd=lambd * eta),
    lb=-100, ub=100,
)
demo.set_objective_norm(zk - z_star, 2)
```

Internally, `fixed_point` creates a new iterate $z^\star$ with the given bounds, applies the
operator to obtain $T(z^\star, x)$, and adds the equality constraint $z^\star = T(z^\star, x)$. The returned
expression then enters the objective, as above, which maximizes the distance between the
$k$th iterate and the fixed point. The bounds on $z^\star$ must hold for all parameters in the
declared parameter set. They can be overestimated, but the tightness of both the MIP
formulation and the SDP relaxation depends on reasonable estimates.

## [Code generation](../api/codegen.md)

The same specification that drives verification also drives code generation, so a method
that has been verified needs no restructuring to be compiled: the program generated from is
the program verified. The user records the algorithm exactly as above, keeps the iterate
sequence, and passes it to
[`codegen`](../api/verifier.md#fomtools.Verifier.codegen), which extracts an
intermediate representation of the run, hands the resulting specification to an LLM coding
agent, and iterates the generated code against a Python reference oracle until it is
numerically verified. The emitted directory is a self contained C project with an
OSQP style ABI, all numeric data, a `Makefile`, and a small `ctypes` wrapper.

This call requires the `codegen` extra and an authenticated coding agent driver.
Set `FOMTOOLS_CODEGEN_AGENT=1` or pass an explicit `driver=`. See the
[code generation guide](codegen.md#generating-the-solver).

```python
# the same trajectory that was verified, now compiled
result = VP.codegen("out/fista", iterates=s, name="fista")
assert result.success
```

Generating the run would emit one block per iteration, so before unrolling
anything the extractor pulls out a single iteration and asks whether repeating it reproduces
the recording. When it does, the compact loop is what gets emitted, and the check that
decides to roll is itself the proof that rolling was safe. Iteration varying coefficients,
such as the FISTA momentum above, are handled without any special syntax, since the recorded
tape simply gives each iteration its own number. Affine operators are factorized once at
generation time rather than inverted, and structured matrices are emitted as sparse
operations, so the generated solver is fast before any optimization pass runs. The
[code generation guide](codegen.md) covers rolling, tabulated coefficient schedules,
termination criteria, and the emitted project layout in full.

## [Implementation notes](../api/canonicalizer.md)

The package separates the user facing front end, including expressions, steps, objectives, and
fixed point encoding, from the backend canonicalizers and solvers. All canonicalizers
implement a common interface: accept variables with bounds, add constraints, set an
objective norm, solve, and retrieve solution values. Adding a new backend therefore amounts
to implementing this interface, without touching the upstream pipeline. After a solve, the
optimal value of any expression can be queried, including nonleaf expressions, which the
package reconstructs from the leaf solution through the stored affine representation. All
coefficient matrices are stored in sparse format using `scipy.sparse`, and interval arithmetic
bound propagation through affine maps is lazy, so bounds are recomputed only when a
canonicalizer consumes them.
