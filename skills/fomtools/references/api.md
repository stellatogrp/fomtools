# FOMTools API Reference

## Verifier

The main orchestrator class. Manages iterates, parameters, steps, constraints,
objectives, and solving.

### Constructor

```python
Verifier(
    theory_func=None,
    method="mip",          # "mip", "sdp", or "simulator"
    solver="highs",       # "gurobi", "highs", "clarabel", "cuclarabel"
    obbt_solver="auto",   # "auto" follows solver if Gurobi/HiGHS, else off; None disables
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
    obbt_params=None,      # solver parameters for OBBT (default: inherit verbosity)
    verbose=False,         # print solver logs
)
```

### Core Methods

#### `iterate(ndim, lb, ub) -> LinearExpression`
(`add_iterate` is an alias.) Create an iterate variable with dimension `ndim` and bounds `[lb, ub]`.
Bounds can be scalars (broadcast) or arrays.

#### `parameter(ndim, lb, ub) -> LinearExpression`
Create a parameter variable with dimension `ndim` and bounds `[lb, ub]`.
Parameters represent uncertain problem data. With `method="simulator"`, pass
`samples` of shape `(num_samples, ndim)`. `add_param` is a deprecated alias.

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

Accessed via `vp.step.<name>(arg, **kwargs)` or directly as `vp.<name>(...)`:

- `vp.step.relu(arg, proj_ranges=None)` -- ReLU / projection onto non-negative orthant
- `vp.step.soft_threshold(arg, lambd=t)` -- soft thresholding
- `vp.step.satlin(arg, l=l, u=u)` -- box projection
- `vp.step.implicit_linear(arg, output_mat=M, weight=None, bias=None)` -- implicit affine solve `M w = weight @ arg + bias`

All step operators return a `LinearExpression`.

Alternatively: `vp.run_step(arg, step_cls, relax_binary_vars=False, **kwargs)`.

### Proximal Operators

```python
vp.prox(func, arg, gamma=1.0) -> LinearExpression
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
vp.quadratic_program(P, q, A=None, b=None, C=None, d=None,
                     dual_ub=None, primal_bounds=None, nu_bounds=None)
vp.linear_program(q, A=None, b=None, C=None, d=None,
                  dual_ub=None, primal_bounds=None, nu_bounds=None)
vp.standard_form_linear_program(c, A=None, b=None,
                                dual_ub=None, primal_bounds=None, nu_bounds=None)
```

### Objectives and solving

```python
vp.set_objective_norm(expr, ord=np.inf)  # infinity norm; 1 for one-norm, 2 for squared two-norm
vp.maximize(ft.norm(expr, np.inf))       # equivalent; also accepts ft.sum_squares(expr)
vp.set_zero_objective()
vp.fixed_point(n_dim, operator, lb, ub)  # new iterate z_star == operator(z_star)
result = vp.solve(convex_hull_cuts=False, cut_rounds=1, tighten_objective_bounds=None,
                  warm_start=True, verbose=None)  # native solver options also accepted
vp.set_start_samples({param: samples})   # MIP start from sampled worst case
```

MIP returns a finite best upper bound even at a time limit when available, and
raises `RuntimeError` otherwise. The solved copy retains the native status and
incumbent separately. SDP returns a dual objective only after an accurate optimal
termination. Simulator returns sampled residual arrays. See `docs/index.md` for
numerical limitations and result inspection.

### Post-processing

`vp.post_process(target_expr, ref_expr, residuals)` uses the triangle inequality.
Supply upper bounds on every successive infinity-norm difference along the path
from reference to target over the same family. Do not use sample maxima,
incumbents, squared norms, or a bound only for the last step.

### Solution Values

```python
vp.get_solution_value(iterate) -> np.ndarray
```

### Code Generation

```python
vp.codegen(out_dir, iterates=None, config=None, driver=None,
           state=None, params=None, name="fom",
           code_optimization=False, profile=False, roll=True, param_values=None)
    -> CodegenResult
```

```python
vp.codegen_rolled(out_dir, next_state, config=None, driver=None,
                  state=None, params=None, name="fom",
                  code_optimization=False, profile=False,
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
- `ClarabelSolver` -- for SDP verification (Clarabel)
- `CuClarabelSolver` -- GPU-accelerated SDP (cuClarabel)
