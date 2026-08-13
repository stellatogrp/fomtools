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
directly.  Code generation comes later.

The same control problem is posed two ways, one per subdirectory.

| | variables | constraint rows | system matrix | per iteration |
|---|---:|---:|---|---:|
| `condensed/` | 40 | 40 | dense, 100% filled | 6.3 us |
| `uncondensed/` | 1240 | 1240 | sparse, 0.99% filled, factor 26739 | 36.5 us |

**Condensed** eliminates the states through the dynamics, leaving a small dense
program in the input moves alone, with the identity as its constraint matrix and
the measured state folded into the linear cost.  **Uncondensed** keeps the
states as variables and the dynamics as constraints: thirty-one times as many
variables, but every matrix banded, and the measured state entering through the
constraint bounds instead.  Both minimize the same function of the input
sequence and have the same minimizing first input, checked to 4e-8 in
`compare.py`, so any difference in how they fly is a property of the algorithm
rather than of the controller.

## Running it

Everything runs as a module from this directory.

```
python -m condensed.check_against_fomtools    # the controller is the recorded algorithm
python -m condensed.run_budget_sweep          # flights, table, figures, animation
python -m uncondensed.check_against_fomtools
python -m uncondensed.run_budget_sweep
python -m compare                             # the two at equal wall-clock budgets
```

The two studies write to their own `out/`; `compare.py` writes to `out/` here.

## What the comparison says

Each formulation on its own answers "how many iterations does this need".  The
question a controller faces is how well it flies given a fixed slice of time,
and since the two do not cost the same per iteration, that needs both numbers.
Converting a budget into an iteration count separately for each, over an eight
second flight:

| budget | condensed | | uncondensed | |
|---:|---:|---|---:|---|
| 0.2 ms | 31 it | crashed | 5 it | crashed |
| 0.5 ms | 78 it | crashed | 13 it | 1.79 m |
| 1 ms | 157 it | crashed | 27 it | 2.08 m |
| 2 ms | 315 it | 0.43 m | 54 it | 2.21 m |
| 5 ms | 789 it | 0.11 m | 137 it | 1.16 m |
| 10 ms | 1578 it | 0.12 m | 274 it | 0.22 m |

Three things come out of this.

**Sparsity does its job, and it is still not enough.**  Thirty-one times the
variables cost only six times the time per iteration, which is the sparse
factorization working as advertised: the factor holds 26739 entries where a
dense one of that size would hold 769000.  But six times is six times, and both
formulations need a comparable number of iterations to converge, so the
condensed one is ahead at every budget.  Eliminating the states is worth more
here than exploiting the sparsity that not eliminating them leaves behind.

**The advantage is a property of this problem's size.**  The condensed Hessian
is 100% filled at any horizon, so its cost grows with the square of the number
of input moves while the uncondensed one grows linearly in the horizon.  Ten
input moves is far inside the range where dense wins.  The crossover is a
horizon question, not a formulation question, and this experiment sits on one
side of it by construction: the move blocking that keeps the condensed program
at 40 variables is what makes it the better choice.

**Starved, the uncondensed controller fails more gracefully.**  Below about one
millisecond the condensed controller crashes outright while the uncondensed one
stays airborne at two metres of error.  The large penalty its equality rows
carry keeps the predicted trajectory roughly consistent with the dynamics even
when the solve is nowhere near converged, so the first input it returns is
merely sluggish rather than wrong.  The condensed formulation has substituted
the dynamics away and has no such protection: an under-converged iterate is an
arbitrary input sequence.

## The setup

**Plant.**  The hover linearization of a quadcopter from the OSQP documentation,
twelve states and four rotor thrusts, taken verbatim from the verification
experiments in `mip_algo_verify` so that both analyze the same vehicle.  It is
published at a 0.1 s sampling period; `common/quadcopter.py` recovers the
underlying continuous-time model and re-discretizes at 10 ms.  Every eigenvalue
is real and positive, so the matrix logarithm is real and the recovery is exact
to machine precision.  The plant and the controller's prediction model are the
same, so there is no model mismatch and the only source of tracking error is the
finite iteration budget.

**Move blocking.**  The input is held constant over blocks of ten sampling
instants in both formulations.  A one second lookahead at 10 ms is 100 steps,
which condensed would turn into 396 variables and uncondensed into 1596;
blocking gives 40 and 1240 at the same lookahead.  For the condensed program
this is what keeps the interpreted iteration dominated by per-operation
overhead, which is exactly the cost that generated code removes.

**Algorithm.**  Equation (16):

```
w    = clip(v, l, u)
zeta = solve(P + sigma I + M' R M,  sigma zeta - q + M' R (2 w - v))
v    = v + M zeta - w
```

with `M = I` and a scalar `R = rho` in the condensed case.  `rho = 0.1` is
OSQP's default penalty and `sigma = 1e-4` is the regularization used in the
verification experiments.  The system matrix is factorized once when the problem
is built, matching what generated code does with a factorization computed at
generation time.  The iterate is cold started at every sampling instant,
matching the generated interface, which resets its state on every parameter
update.

**Reference.**  A figure eight of amplitude 1 m and period 2.25 s with a slow
climb, flown from the origin so the flight opens with a capture transient.  The
defaults sit at the edge of what the vehicle can fly: the thrusts are on their
bounds for roughly three quarters of the flight.  This matters — a gentler
reference leaves the box slack, the program becomes effectively unconstrained,
and the iteration count stops mattering at all.

## Files

| file | contents |
|---|---|
| `common/quadcopter.py` | the model, its cost and constraint data, re-discretization |
| `common/reference.py` | the figure eight, and a random B-spline generator |
| `common/simulate.py` | the `Controller` protocol, the closed loop, crash detection, timing |
| `common/plots.py` | 3-D panels, the still comparison, the animation, error against time |
| `common/sweep.py` | the budget sweep both studies run |
| `<study>/mpc.py` | that formulation's problem assembly |
| `<study>/admm.py` | that formulation's iteration |
| `<study>/controller.py` | the adapter the shared simulator flies |
| `compare.py` | the two formulations at equal wall-clock budgets |

## A note on code generation

The two formulations are not equally ready to be generated as C.  The condensed
one confines the parameter to the linear cost, so its saturated-linear step has
constant bounds, which is the only form `fomtools.codegen` accepts: its
intermediate representation stores those bounds as names of constant arrays
(`ir.py`, `ActivationOp.lower`), and `extract.py:_satlin_arrays` raises
`NonStationaryFOMError` on a bound that is a `LinearExpression`.  The
uncondensed one puts the measured state into the bounds, so a *deployed*
uncondensed solver would need that extension.

This does not affect the experiment.  The simulation supplies the bounds as
numbers refreshed each instant, never as symbolic parameters, and
`uncondensed/check_against_fomtools.py` records the iteration with the bounds
frozen at one representative state, which is enough to establish that the
hand-written controller, the recorded steps, and the extracted representation
are the same map.  The restriction is worth knowing because §6.4 of the paper
currently describes the uncondensed form.
