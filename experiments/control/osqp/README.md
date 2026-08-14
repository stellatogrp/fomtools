# Quadcopter closed-loop control experiment

A quadcopter tracks a reference trajectory under a model predictive controller
whose quadratic program is solved by the ADMM iteration of the paper, equation
(16).  The controller is given a fixed number of iterations per sampling
instant, and nothing else changes between runs.  Too few, and the quadcopter
loses the trajectory and flies away.

The experiment is about *implementations*.  A NumPy iteration, an emitted C
iteration, and an agent-optimized C iteration each buy a different number of
iterations inside the same wall-clock budget, so the same controller flies
cleanly, wobbles, or crashes depending only on how fast its inner loop runs.  The
iteration count is swept directly first, to calibrate what the controller needs;
then code generation supplies the three implementations and the budget decides
how many each of them gets.

The same control problem is posed two ways, one per subdirectory.

| | variables | constraint rows | system matrix | NumPy per iteration |
|---|---:|---:|---|---:|
| `condensed/` | 40 | 40 | dense, 100% filled | 6.06 us |
| `uncondensed/` | 1240 | 1240 | sparse, 0.99% filled, factor 26739 | 37.39 us |

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

Everything runs as a module from this directory (`experiments/control/osqp`).
The plant, the simulator, the plots, and the code-generation pipeline live one
level up in `../common`, shared with the `rapdhg` study; each study package puts
that directory on the path itself, so no environment variable is needed.

```
python -m condensed.check_against_fomtools    # the controller is the recorded algorithm
python -m condensed.run_budget_sweep          # flights against the iteration count
python -m uncondensed.check_against_fomtools
python -m uncondensed.run_budget_sweep
python -m compare                             # the two at equal wall-clock budgets

FOMTOOLS_CODEGEN_AGENT=1 python -m condensed.generate     # emit both C solvers
python -m condensed.calibrate                 # what one iteration costs, per implementation
python -m condensed.run_implementations       # flights against the compute budget
```

The last three take a `<study>.` prefix of either formulation.  Each study writes
to its own `out/`; `compare.py` writes to `out/` here.  The generated solvers are
reused on a later run while they still pass `make verify`, so the agent phases
are paid for once.

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
published at a 0.1 s sampling period; `../common/quadcopter.py` recovers the
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
| `../common/quadcopter.py` | the model, its cost and constraint data, re-discretization |
| `../common/reference.py` | the figure eight, and a random B-spline generator |
| `../common/simulate.py` | the `Controller` protocol, the closed loop, crash detection, timing |
| `../common/plots.py` | 3-D panels, the still comparison, the animation, error against time |
| `../common/sweep.py` | the budget sweep both studies run |
| `../common/csolver.py` | a generated C solver, built, timed, and callable |
| `../common/codegen_driver.py` | emits the translated and the optimized solver, keeping both |
| `<study>/mpc.py` | that formulation's problem assembly |
| `<study>/admm.py` | that formulation's iteration |
| `../common/recording.py` | parameter boxes and the recording container |
| `../common/study.py` | the code-generation pipeline both formulations run |
| `<study>/controller.py` | the adapter the shared simulator flies |
| `<study>/recording.py` | that formulation's iteration, recorded on a `Verifier` |
| `<study>/study.py` | the adapter that makes it a `Study` |
| `<study>/generate.py` | runs both agent phases |
| `<study>/calibrate.py` | what one iteration costs, per implementation |
| `<study>/run_implementations.py` | the three implementations at equal compute budgets |
| `compare.py` | the two formulations at equal wall-clock budgets |

## Code generation

Both formulations are generated as C and flown, in three implementations of one
algorithm: NumPy, the solver as first emitted, and the same solver after the
optimization phase.  Every arm is checked against the others to 1e-10, and flown
at a common iteration count they agree to 4e-10, so a difference between their
flights is a difference in speed and nothing else.

| | multiply-adds per iteration | NumPy | emitted C | optimized C | emission |
|---|---:|---:|---:|---:|---:|
| `condensed/` | 2000 | 6.06 us | 2.88 us | 2.67 us | **2.1x faster** |
| `uncondensed/` | 65780 | 37.39 us | 71.92 us | 64.32 us | **1.9x slower** |

The optimization phase is the disappointing column, and the reason is the same
in both rows: 91% and 88% of the emitted runtime sits inside `fom_lin_solve_t1`
in `fom_linsolve.c`, a deterministic file restored before every verification and
therefore not the agent's to edit.  That is a property of the *algorithm*, not of
the generator.  The `../rapdhg` study next door runs a matrix-free method on this
same condensed program, where the emitted iteration contains no linear solve at
all, and the optimization phase there buys **17.9x** instead of 1.08x.

**Emission pays exactly where the interpreter is the cost.**  The condensed
iteration runs 2000 multiply-adds in 6.1 us, which is 0.3 G/s: hardly any of that
is arithmetic, it is per-operation overhead on vectors of forty numbers, and
removing it is what code generation does.  The uncondensed iteration runs 65780
in 37 us, which is 1.8 G/s of compiled SuperLU and CSR kernels with nothing left
to recover — its per-solve cost outside the loop measures 1.1 us.  Generated C
would have to beat 1.8 G/s in scalar CSC loops merely to break even, and it does
not: the emitted triangular solve takes 53.8 us where SuperLU takes 21.8, because
the emitter writes a scalar loop and SuperLU works on dense blocks within
supernodes.

So the choice of formulation decides whether code generation helps at all.
Eliminating the states buys a small dense program *and* the conditions under
which emitting C pays; keeping them buys sparsity that NumPy already exploits.

**In the condensed study that produces the experiment.**  Between a 0.6 and a
1.0 ms budget the interpreted controller crashes the quadcopter and both
generated solvers fly it:

| budget | NumPy | emitted C | optimized C |
|---:|---|---|---|
| 0.60 ms | 99 it, crashed | 208 it, 1.92 m | 224 it, 1.29 m |
| 0.80 ms | 132 it, **crashed** | 277 it, **0.73 m** | 299 it, **0.56 m** |
| 1.00 ms | 165 it, crashed | 347 it, 0.24 m | 374 it, 0.16 m |

**The optimization phase is capped on both, and not by the agent.**  It bought
1.08x on the condensed program and 1.12x on the uncondensed one.  Both
translation agents reported the same cause without being asked: 91% and 88% of
runtime inside `fom_lin_solve_t1`, which lives in `fom_linsolve.c`, a
deterministic file restored before every verify and therefore off limits.  Two
problems thirty-one times apart in size, one with a dense 820-nnz factor and one
with a sparse 18679-nnz factor, the same ceiling.  What limits the optimization
phase on this family is the emitted linear solve rather than what the agent can
find — the condensed agent wrote 397 lines and discovered that every coefficient
block is a scalar multiple of the identity, for 8%.

One formulation needs a rewrite to be generatable at all.  The uncondensed one
puts the measured state into the constraint bounds, which `fomtools.codegen`
cannot express — its intermediate representation stores saturated-linear bounds
as names of constant arrays (`ir.py`, `ActivationOp.lower`), and
`extract.py:_satlin_arrays` raises `NonStationaryFOMError` on a bound that is a
`LinearExpression` — but the identity `sat_[l,u](y) = l + sat_[0,u-l](y-l)` moves
the parameter into the expression the step is applied to, where it is accepted.
The rewrite is exact and, in this program, nearly free: the width is constant on
every row.  See `uncondensed/recording.py`.

Details, profiles and the flights are in each study's own README.
