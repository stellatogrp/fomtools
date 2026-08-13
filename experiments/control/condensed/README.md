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
python -m condensed.check_against_fomtools
python -m condensed.run_budget_sweep
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
6.2 us here, so a 10 ms budget buys roughly 1600 of them.

Two things worth knowing about the table.  Tracking error is very slightly
*worse* at 5000 iterations than at 500: an under-converged iterate is not a
worse control law in any monotone sense, it is a different one, and here it
happens to lag marginally less.  And the crashed rows have no meaningful error,
since a diverging flight leaves the region where the linear model describes a
quadcopter at all.

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
