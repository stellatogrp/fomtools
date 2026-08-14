# Condensed formulation

The states are eliminated through the dynamics, leaving a box-constrained
quadratic program in the input moves alone,

```
minimize    (1/2) z' P z + q(x, r)' z
subject to  l <= z <= u,
```

with 40 variables at a one second lookahead.  Three things follow from the
elimination.  The constraint matrix is the identity, so the ADMM iteration
performs no matrix-vector products beyond its linear solve.  The bounds are the
constant thrust box.  And the parameter enters only through the linear cost,
`q = G_x x - G_r r`, affine in the measured state and the reference window.

That last property is what makes this formulation suitable for code generation:
the saturated-linear step has constant bounds, which is the only form the
extraction accepts, and the system matrix `P + sigma I + rho I` is fixed and
factorized once.

The price is that `P` is 100% filled.  Condensing trades sparsity for size, and
at 40 variables that is a good trade — see the comparison in the parent README.

## Running it

```
python -m condensed.check_against_fomtools    # the controller is the recorded algorithm
python -m condensed.run_budget_sweep          # flights against the iteration count
FOMTOOLS_CODEGEN_AGENT=1 python -m condensed.generate      # emit both C solvers
python -m condensed.calibrate                 # what one iteration costs, per implementation
python -m condensed.run_implementations       # flights against the compute budget
```

Outputs land in `out/`:

| file | what it is |
|---|---|
| `sweep.csv` | one row per iteration budget: tracking error, crash, saturation |
| `rollouts.npz` | the flown trajectories, for replotting without re-flying |
| `comparison.pdf` | side-by-side 3-D panels at three illustrative budgets |
| `flight.gif` | the same comparison, animated |
| `error.pdf` | tracking error against time |
| `parameter_box.npz` | the box of linear costs a flight visits (written by the check) |
| `implementations.csv` | one row per implementation per compute budget |
| `implementations_budget.pdf` | tracking error against the budget, one curve per implementation |
| `implementations_flight.pdf` | the three implementations at the headline budget |
| `implementations_error.pdf` | tracking error against time at that budget |
| `implementations.gif` | the same three flights, animated |
| `codegen/{emitted,optimized}/` | the generated solvers, reused while they still verify |

## Calibration

Eight second flights, 10 ms sampling, error in metres.  "Steady" excludes the
first two seconds, which is the climb from the origin up to the trajectory.

| iterations | outcome | steady error | thrusts on the box |
|---:|---|---:|---:|
| 10 | crashed at 3.1 s | | |
| 25 | crashed at 4.0 s | | |
| 50 | crashed at 3.9 s | | |
| 100 | crashed at 5.3 s | | |
| 250 | flew | 0.93 m | 84% |
| 500 | flew | 0.11 m | 85% |
| 1000 | flew | 0.12 m | 74% |
| 2500 | flew | 0.12 m | 71% |
| 5000 | flew | 0.12 m | 68% |

Three regimes, and the figures use one budget from each: lost below 250, holding
on badly at 250, flying the curve from 500.  A NumPy iteration costs about
6.1 us here, so a 10 ms budget buys roughly 1650 of them.

Two things worth knowing about the table.  Tracking error is very slightly
*worse* at 5000 iterations than at 500: an under-converged iterate is not a
worse control law in any monotone sense, it is a different one, and here it
happens to lag marginally less.  And the crashed rows have no meaningful error,
since a diverging flight leaves the region where the linear model describes a
quadcopter at all.

## Code generation

Nothing has to be rewritten to generate this formulation: the bounds are already
constant and the parameter is already confined to the linear cost, which is the
shape extraction accepts.  What comes out is two body steps over a state of
dimension 80, one 40-long parameter, nine coefficient blocks holding 360 applied entries,
and a dense LDL^T factorization holding 820 — **2000 multiply-adds per
iteration**, against 65780 for the uncondensed program.

### Emission pays here

| implementation | us/iteration | spread | iterations in 10 ms |
|---|---:|---:|---:|
| NumPy | 6.06 | 3.8% | 1650 |
| emitted C | 2.88 | 6.8% | 3472 |
| optimized C | 2.67 | 1.7% | 3741 |

All three agree with each other to 1e-10 at every iteration count checked, and
flown for eight seconds at a common count their trajectories differ by 4e-10, so
this is one algorithm in three implementations and nothing else.

Emission is worth 2.1x, where on the uncondensed program it *cost* a factor of
two.  The reason is where the time goes.  NumPy runs 2000 multiply-adds in 6.1 us,
which is 0.3 G/s: almost none of that is arithmetic, it is per-operation
interpreter overhead on vectors of forty numbers, and removing it is what code
generation does.  The uncondensed iteration runs 65780 in 37 us, which is 1.8 G/s
of already-compiled sparse kernels with nothing left to recover.

### Flights against the compute budget

Eight second flights.  The sampling period is 10 ms, the ceiling on any budget.

| budget | NumPy | | emitted C | | optimized C | |
|---:|---:|---|---:|---|---:|---|
| 0.40 ms | 66 it | crashed | 138 it | crashed | 149 it | crashed |
| 0.60 ms | 99 it | crashed | 208 it | 1.92 m | 224 it | 1.29 m |
| 0.80 ms | 132 it | **crashed** | 277 it | **0.73 m** | 299 it | **0.56 m** |
| 1.00 ms | 165 it | crashed | 347 it | 0.24 m | 374 it | 0.16 m |
| 1.50 ms | 247 it | 0.95 m | 520 it | 0.11 m | 561 it | 0.11 m |
| 10.0 ms | 1650 it | 0.12 m | 3471 it | 0.12 m | 3741 it | 0.12 m |

Between 0.6 and 1.0 ms the interpreted controller crashes the quadcopter and both
generated solvers fly it.  That band is the experiment: the algorithm, the
problem, the reference and the cold start are identical across the three columns,
and the only thing that differs is how fast the inner loop runs.  Above 1.5 ms
every implementation has enough iterations and the distinction disappears.

### The optimization phase is capped, and not by the agent

Translation is worth 2.1x; the optimization phase on top of it is worth 1.08x.
That is not for want of effort — the agent rewrote the solver from 136 lines to
397, with a setup-time probe that recovers every coefficient block as a scalar
multiple of the identity and a fused fast path guarded by a fallback.  The
translation agent had already reported the ceiling:

> Profiling shows 91% of the time is inside the provided `fom_lin_solve_t1` LDL^T
> (820 nnz, two triangular solves x 20 iterations), which is off-limits to
> change; my code is the remaining ~9%.

`fom_linsolve.c` is a deterministic file, restored before every verify, so the
optimization phase cannot touch the thing that dominates.  The uncondensed study
hit the same wall at 88%, on a problem thirty-one times larger with a sparse
factor instead of a dense one.  Two points, same ceiling: what limits the
optimization phase on this family is the emitted linear solve, not the agent.

## Checks

`check_against_fomtools.py` runs the same data through three executions: the
NumPy controller, `run_faithful` on the recorded step objects, and
`ReferenceOracle` on the extracted intermediate representation that code
generation lowers to C.  The three agree to about 2e-11, which is the
precondition for generating a solver from this recording.

It also compares the converged iterate against OSQP *by objective value* rather
than by iterate.  Most coordinates of the solution sit exactly on the thrust box
and the remaining subspace is poorly conditioned, so the minimizer is not
numerically well determined: OSQP and an interior-point solver disagree with
each other by more than either disagrees with the converged ADMM iterate, while
all three agree on the objective to about 1e-13 relative.
