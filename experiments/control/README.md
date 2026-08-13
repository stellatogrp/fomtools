# Quadcopter closed-loop control experiment

A quadcopter tracks a reference trajectory under a model predictive controller
whose quadratic program is solved by the ADMM iteration of the paper, equation
(16).  The controller is given a fixed number of iterations per sampling
instant, and nothing else changes between runs.  Too few, and the quadcopter
loses the trajectory and flies away.

The experiment this feeds is about *implementations*.  A NumPy iteration, an
emitted C iteration, and an agent-optimized C iteration each buy a different
number of iterations inside the same wall-clock budget, so the same controller
flies cleanly, wobbles, or crashes depending only on how fast its inner loop
runs.  This directory is the first stage: the problem, the algorithm, the
simulation, and the pictures, calibrated by sweeping the iteration count
directly.  Code generation and the wall-clock budget sweep come later.

## Running it

Everything is run from this directory.

```
python check_against_fomtools.py   # the controller is the algorithm fomtools records
python run_budget_sweep.py         # the flights, the table, the figures, the animation
```

`run_budget_sweep.py` writes to `out/`:

| file | what it is |
|---|---|
| `sweep.csv` | one row per iteration budget: tracking error, crash, saturation |
| `rollouts.npz` | the flown trajectories, for replotting without re-flying |
| `comparison.pdf` | side-by-side 3-D panels at three illustrative budgets |
| `flight.gif` | the same comparison, animated |
| `error.pdf` | tracking error against time |
| `parameter_box.npz` | the box of linear costs a flight visits (written by the check) |

Useful flags: `--budgets`, `--figure-budgets`, `--duration`, `--no-gif`.

## The setup

**Plant.**  The hover linearization of a quadcopter from the OSQP documentation,
twelve states and four rotor thrusts, taken verbatim from the verification
experiments in `mip_algo_verify` so that both analyze the same vehicle.  It is
published at a 0.1 s sampling period; `quadcopter.py` recovers the underlying
continuous-time model and re-discretizes at 10 ms.  Every eigenvalue is real and
positive, so the matrix logarithm is real and the recovery is exact to machine
precision.  The plant and the controller's prediction model are the same, so
there is no model mismatch and the only source of tracking error is the finite
iteration budget.

**Problem.**  States are eliminated, leaving a box-constrained quadratic program
in the input sequence.  The constraint matrix is then the identity, the bounds
are the constant thrust box, and the parameter enters only through the linear
cost.  That last property is what makes the family suitable for code generation:
the saturated-linear step has constant bounds, which is the only form the
extraction accepts, and the system matrix is fixed and factorized once.

The input is held constant over blocks of ten sampling instants.  A one second
lookahead at 10 ms is 100 steps and would give 396 variables; blocking gives 40
at the same lookahead.  This is not cosmetic.  At 40 variables an interpreted
iteration is dominated by per-operation overhead, which is exactly what
generated code removes; at 396 both implementations spend their time inside the
same dense triangular solve and the difference between them largely disappears.

**Algorithm.**  Equation (16), with the constraint matrix equal to the identity:

```
w = clip(v, l, u)
z = solve(P + sigma I + rho I,  sigma z - q + rho (2 w - v))
v = v + z - w
```

`rho = 0.1` is OSQP's default penalty and `sigma = 1e-4` is the regularization
used in the verification experiments; neither is tuned to this instance.  The
system matrix is factorized once when the problem is built, matching what the
generated code does with a factorization computed at generation time.  The
iterate is cold started at every sampling instant, matching the generated
interface, which resets its state on every parameter update.

**Reference.**  A figure eight of amplitude 1 m and period 2.25 s with a slow
climb, flown from the origin so the flight opens with a capture transient.  The
defaults sit at the edge of what the vehicle can fly: the thrusts are on their
bounds for roughly three quarters of the flight.  This matters — a gentler
reference leaves the box slack, the program becomes effectively unconstrained,
and the iteration count stops mattering at all.

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

Three regimes, and the figures use one budget from each: lost below 250,
holding on badly at 250, flying the curve from 500.  A NumPy iteration costs
about 6.0 us here, so a 10 ms budget buys roughly 1660 of them; converting a
time budget into an iteration budget for each implementation is what stage
three does with that number.

Two things worth knowing about the table.  Tracking error is very slightly
*worse* at 5000 iterations than at 500: an under-converged iterate is not a
worse control law in any monotone sense, it is a different one, and here it
happens to lag marginally less.  And the crashed rows have no meaningful error,
since a diverging flight leaves the region where the linear model describes a
quadcopter at all.

## Files

| file | contents |
|---|---|
| `quadcopter.py` | the model, its cost and constraint data, re-discretization |
| `mpc.py` | condensing, move blocking, the parametric linear cost, the factorization |
| `reference.py` | the figure eight, and a random B-spline generator for later |
| `admm.py` | the iteration, the scaled residual, and an OSQP reference solve |
| `simulate.py` | the closed loop and crash detection |
| `plots.py` | 3-D panels, the still comparison, the animation, error against time |
| `run_budget_sweep.py` | the driver |
| `check_against_fomtools.py` | the controller against the recorded and extracted algorithm |

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
