# FOMTools API Reference

## Verifier

The main orchestrator class. Manages iterates, parameters, steps, constraints,
objectives, and solving.

### Constructor

```python
Verifier(
    theory_func=None,
    method="mip",          # "mip", "sdp", or "simulator"
    solver="gurobi",       # "gurobi", "highs", "clarabel", "cuclarabel"
    obbt_solver="gurobi",  # solver for bound tightening; None to disable
    params=None,           # solver-specific parameters dict
    # SDP-only cut flags (no effect for method != "sdp"):
    rlt_cuts=False,
    triangle_cuts=False,
    ineq_product_cuts=True,
    ineq_box_cuts=True,
    ineq_square_cuts=True,
    relu_envelope_cuts=False,
    rlt_all_pairs=False,
    full_psd_cone=False,
    cross_cut=False,
    subgrad_mono_cut=False,
)
```

### Core Methods

#### `add_iterate(dim, lb, ub) -> LinearExpression`
Create an iterate variable with dimension `dim` and bounds `[lb, ub]`.
Bounds can be scalars (broadcast) or arrays.

#### `add_param(dim, lb, ub) -> LinearExpression`
Create a parameter variable with dimension `dim` and bounds `[lb, ub]`.
Parameters represent uncertain problem data.

#### `add_param_stack(param_list) -> LinearExpression`
Stack a list of `LinearExpression` parameters into one.

#### `iterate_trajectory(length, collapse_expr="auto")`
Create a list of `length` slots for iterates. `collapse_expr="auto"` detects
accumulating methods (like Douglas-Rachford) where expression support grows
and collapses them to keep linear growth.

#### `add_constraint(constr)`
Add a `Constraint` to the verification problem.

#### `complementarity(expr_a, expr_b)`
Add a complementarity constraint: `expr_a >= 0, expr_b >= 0, expr_a * expr_b == 0`.

#### `sum_squares(expr) -> SumSquares`
Create a sum-of-squares expression for use in constraints.

### Step Operators

Accessed via `vp.step.<name>(arg, **kwargs)`:

- `vp.step.relu(arg)` -- ReLU / projection onto non-negative orthant
- `vp.step.soft_threshold(arg, lambd=t)` -- soft thresholding
- `vp.step.saturated_linear(arg, lb, ub)` -- box projection
- `vp.step.affine(arg, output_mat=M)` -- linear transformation
- `vp.step.implicit_linear(arg, output_mat=M)` -- implicit linear solve

All step operators return a `LinearExpression`.

Alternatively: `vp.run_step(arg, step_cls, relax_binary_vars=False, **kwargs)`.

### Proximal Operators

```python
vp.prox(func, arg, step_size=1.0) -> LinearExpression
```
Compute the proximal operator of `func` at `arg`. `func` is from
`fomtools.proxable_functions` (e.g., `NormL1`).

### Projection

```python
vp.project(proj_set, arg) -> LinearExpression
```
Project `arg` onto `proj_set`. A parameterless `Set` subclass can be passed
as a bare class (e.g., `NonNegative`).

### Problem Builders

These convenience methods set up standard optimization problems on the verifier:

```python
vp.quadratic_program(P, q, A=None, b=None, lb=None, ub=None) -> LinearExpression
vp.linear_program(c, A=None, b=None, lb=None, ub=None) -> LinearExpression
vp.standard_form_linear_program(c, A, b, n) -> LinearExpression
```

### Objectives

```python
vp.set_infinity_norm_objective(expr)      # maximize ||expr||_inf
vp.set_two_norm_objective(expr)           # maximize ||expr||_2^2
vp.set_one_norm_objective(expr)           # maximize ||expr||_1
vp.set_objective_norm(expr, norm="inf")   # generic: "inf", "two", "one"
vp.set_zero_objective()                   # set objective to zero
```

### Solving

```python
result = vp.solve(**kwargs) -> float
```
Solve the verification problem. Returns the optimal value. Keyword arguments
are forwarded to the canonicalizer (e.g., `huchette_cuts=True` for MIP).

### Post-Processing

```python
vp.post_process(target_expr, ref_expr, residuals)
```
Tighten bounds after solving. Call after each `vp.solve()`.

### Solution Values

```python
vp.get_solution_value(iterate) -> np.ndarray
```

### Code Generation

```python
vp.codegen(out_dir, iterates=None, config=None, driver=None,
           state=None, params=None, name="fom",
           run_phase_b=False, profile=False, roll=True, param_values=None)
    -> CodegenResult
```

```python
vp.codegen_rolled(out_dir, next_state, config=None, driver=None,
                  state=None, params=None, name="fom",
                  run_phase_b=False, profile=False,
                  iterates=None, param_values=None)
    -> CodegenResult
```

### Neural Networks

```python
vp.neural_network(expr, layers) -> LinearExpression
```
Verify a small neural network as a FOM. `layers` is a list of
`LinearLayer`/`ReLULayer` from `fomtools.nn`.

## LinearExpression

Symbolic linear combination of variables/params. Supports:

- `@` operator: matrix-vector multiplication (`M @ expr`)
- Arithmetic: `+`, `-`, `*` (scalar), `/` (scalar)
- `stack(exprs)`: vertically stack expressions
- Indexing: `expr[i]`, `expr[i:j]`
- Bounds: `expr.lb`, `expr.ub` (propagated via interval arithmetic)

## Constraints

```python
from fomtools.constraint import EqualityConstraint, InequalityConstraint

vp.add_constraint(EqualityConstraint(lhs, rhs))
vp.add_constraint(InequalityConstraint(lhs, rhs))  # lhs <= rhs
```

Or via operator overloading on `LinearExpression`:
```python
vp.add_constraint(expr_a == expr_b)
vp.add_constraint(expr_a <= expr_b)
```

## Solvers

- `GurobiSolver` -- for MIP verification (requires Gurobi license)
- `HiGHSSolver` -- open-source MIP solver alternative
- `ClaravelSolver` -- for SDP verification (Clarabel)
- `cuClaravelSolver` -- GPU-accelerated SDP (cuClarabel)
