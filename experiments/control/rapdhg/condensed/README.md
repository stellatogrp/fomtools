# rAPDHG on the condensed quadcopter program

Forty variables, 120 inequality rows, one second of lookahead at a 10 ms
sampling period. The plant, the cost, the reference, and the move blocking come
from `../../common/condensed_qp.py`, shared with the `osqp` study; this study
adds the one state constraint the OSQP example carries and the shared program
leaves out, a bound of 30 degrees on the two tilt angles, and solves the result
with a method with no linear system in it.

## Running it

```
python -m condensed.check_tilt                # which reference the bounded controller can fly
python -m condensed.check_against_fomtools    # the controller is the recorded algorithm
python -m condensed.tuning                    # the K and primal-weight sweep
python -m condensed.generate --max-iters 512  # both C solvers (FOMTOOLS_CODEGEN_AGENT=1)
python -m condensed.extend_tables             # lengthen the schedule tables, no agent
python -m condensed.run_implementations       # the three arms at equal budgets
python -m condensed.run_figures               # the same three on a non-repeating path
python -m condensed.check_render              # the drawn quadcopter points where it accelerates
```

Every flight starts on the reference, at its position and velocity at time
zero, as the learning-to-warm-start flights do; `common.simulate.closed_loop`
takes `on_reference=False` for the older takeoff from the origin.

## The result

| implementation | us/iteration | spread | vs NumPy | vs emitted |
|---|---:|---:|---:|---:|
| NumPy | 13.26 | 0.7% | 1.0x | |
| emitted C (phase A) | 2.87 | 1.2% | **4.6x** | 1.0x |
| optimized C (phase B) | 0.144 | 1.6% | **92x** | **19.9x** |

The per-iteration costs move by a few percent between runs (the generation-time
check measured 12.9, 2.97, and 0.151), which is why the iteration counts below
are not round.

The generated benchmark, five thousand solves at 512 iterations, puts the two
C solvers at 1531 and 74.9 microseconds per solve, 20.4x. Set against the ADMM
study, where emission bought 2.1x and the optimization phase 1.08x. The
difference is not that this agent was cleverer. It is that there was something
left for it to optimize.

Flown for eight seconds on the 4.0 s figure eight from the reference, steady
tracking error in metres, the largest tilt reached in degrees, and the share of
instants more than half a degree over the 30-degree bound. No arm crashes at
any budget.

| budget | NumPy | emitted C | optimized C |
|---:|---|---|---|
| 25 us | 1 it, 1.118 m, 20.6 deg | 8 it, 0.246 m, 23.1 deg | 173 it, **0.004 m**, 31.6 deg, 2.0% |
| **40 us** | 3 it, 0.672 m, 16.6 deg | 13 it, 0.106 m, 27.7 deg | 277 it, **0.003 m**, 31.7 deg, 1.9% |
| 63 us | 4 it, 0.503 m, 17.8 deg | 21 it, 0.039 m, 30.3 deg | 436 it, **0.004 m**, 31.7 deg, 1.9% |
| 160 us | 12 it, 0.128 m, 27.0 deg | 55 it, 0.012 m, 31.3 deg, 1.6% | 1109 it, **0.004 m**, 31.5 deg, 1.7% |
| 400 us | 30 it, 0.038 m, 30.6 deg, 0.6% | 139 it, 0.007 m, 31.5 deg, 1.9% | 2772 it, **0.003 m**, 31.2 deg, 1.5% |
| 1.0 ms | 75 it, 0.013 m, 31.1 deg, 1.4% | 348 it, 0.004 m, 31.7 deg, 2.0% | 6931 it, **0.002 m**, 31.0 deg, 1.2% |
| 2.5 ms | 188 it, 0.004 m, 31.6 deg, 1.9% | 872 it, 0.004 m, 31.5 deg, 1.7% | 17328 it, **0.002 m**, 30.8 deg, 1.1% |

The flight figures are drawn at the **40 us** row, where the three arms are
told apart by eye: three iterations lose the curve, thirteen cut its corners,
and 277 sit on it. At 400 us and above every arm is on the curve and the
panels look alike; that regime is the table's business, not the picture's.

Two things to read off it. The tracking error separates the arms by an order of
magnitude at every budget below a millisecond, and the optimized arm is at its
floor from the first row: 173 iterations in 25 microseconds, a quarter of one
percent of the sampling period. And the tilt column is flat at 30.6 to 31.7
degrees for every flight that tracks at all. The bound is imposed at the end of
each 100 ms input block, and inside a block the angle overshoots it by up to a
degree and a half; the exact solver's flight does the same (30.3 degrees, see
the pre-check table below), so this is the formulation, not the solver. The
low-budget NumPy flights stay under the bound only because they barely turn.

The 2.5 ms row is where the schedule tables stop mattering: the optimized arm
runs 17,328 iterations there, and the tables were extended to 32,768 after
generation for exactly this grid (`extend_tables.py`). The sampling period is
still 10 ms; what this measures is how small a slice of it each implementation
needs.

`run_figures.py` draws the three arms on the quasi-periodic path at the same
40 us budget, where the bound is slack:

| | iterations | steady error | max tilt |
|---|---:|---|---|
| NumPy | 3 | 0.622 m | 17.3 deg |
| emitted C | 13 | 0.070 m | 23.9 deg |
| optimized C | 268 | **0.002 m** | 24.3 deg |

## The tilt bound

The plant is a hover linearization, and the OSQP example it comes from bounds
the two tilt angles by `+-pi/6`. The shared condensed program drops that bound,
and without it the tracking controller does what a linear model lets it: on the
original 2.25 s figure eight the exact solution tilts to 74 degrees, and every
arm flew at 45 to 65. A hover model says nothing at those angles, so this study
puts the bound back, at the last predicted state of every input block,

    -e <= F x + C z <= e,        e = pi/6,

where `F x` is the free response of the two angles from the measured state and
`C z` the forced one. Stacked under the thrust box that is `G = [I; -I; C; -C]`
and `h(x) = [hi; -lo; e - F x; e + F x]`: 120 rows, a right-hand side that
moves with the state, and one real 20 x 40 product per direction where before
every product with `G` was a copy and a sign flip. Only the block ends are
bounded; inside a block the angle can overshoot for at most 100 ms, and the
exact solver's flights show it does so by half a degree.

Two things follow, and both were measured before anything was generated
(`check_tilt.py`, NumPy and Clarabel only).

**The reference has to be one a 30-degree vehicle can fly.** Solved exactly,
the bounded controller on the 2.25 s path lags it by 0.27 m instead of 0.03,
which is the correct behaviour of a tilt-limited drone on a path that asks for
39 degrees of tilt just to follow the horizontal acceleration. That is not a
solver failure, but it is not a tracking experiment either, so the period was
lengthened until the demand peaks at the limit rather than above it.

**A first-order method feels the bound in a way an interior-point solver does
not.** Along a flight on which the bound is active, Clarabel's multipliers on
the tilt rows reach `4e6` (`2e7` on the first block, about `2e3` on the rest)
against `2e5` on the thrust rows, because the objective is a hundred-stage sum
with `||Q|| = 2.6e4`. rAPDHG builds its dual by steps that Theorem 1 caps at
`w / (2 ||G||) = 0.11`, so on a path where the bound has to hold the vehicle
back it never gets there: on the 3.5 s figure eight the relative KKT error sits
at `3e-2` from 469 iterations to 8192, the tilt stays at 40 degrees, and the
flight is identical whether the first block is bounded or not. Neither knob
fixes this. Sweeping `K` from 16 to 128 and `w` from 1 to 16 moves the error
between `2e-2` and `7e-2`; `w` in the hundreds or thousands, or dividing the
objective by `1e2` to `2.6e4`, respects the bound only by giving up tracking
(0.1 to 0.5 m against 0.03). The remedy is the same as above: a path on which
the bound clips the peaks, which at 4.0 s it does.

At K = 32, w = 2, flown from the reference, before the retune below:

| period | reference | exact, unbounded: tilt | exact, bounded: error / tilt | NumPy 469 it: error / tilt / KKT | NumPy 2048 it |
|---:|---|---:|---|---|---|
| 3.0 s | quasi | 44.0 | 0.058 m / 30.6 | 0.016 m / 43.6 / 4.5e-02 | 0.005 m / 44.2 / 4.1e-02 |
| 3.0 s | figure eight | 57.8 | 0.122 m / 30.7 | 0.040 m / 52.3 / 8.9e-02 | 0.007 m / 56.5 / 8.1e-02 |
| 3.5 s | quasi | 31.3 | 0.002 m / 30.3 | 0.007 m / 32.3 / 7.6e-04 | 0.004 m / 32.0 / 6.0e-05 |
| 3.5 s | figure eight | 41.5 | 0.033 m / 30.6 | 0.017 m / 40.4 / 3.4e-02 | 0.005 m / 41.7 / 3.1e-02 |
| **4.0 s** | quasi | 23.3 | 0.001 m / 23.3 | 0.004 m / 24.4 / 1.7e-04 | 0.002 m / 24.0 / 2.9e-05 |
| **4.0 s** | figure eight | 31.0 | 0.003 m / 30.3 | 0.007 m / 31.5 / 2.5e-03 | 0.004 m / 31.6 / 8.0e-05 |

Tilt in degrees, error the steady tracking error, KKT the median relative KKT
error of the solves along the flight. The pattern is the same in every row:
where the unbounded optimum exceeds 30 degrees the first-order solve stalls and
the bound is violated; where it does not, the solve converges and the bound is
kept to within the block-interior overshoot. The study is therefore reported at
a **4.0 s period** on both references, where the figure eight's demand peaks at
31 degrees and the bound is active only at the peaks.

The bound is opt-in in `build_qp` (`tilt_limit=None` is the shared default), so
the `osqp` study still generates from the published unbounded program.

## The trajectories

Both references are Lissajous curves at a **4.0 s period**, and the difference
between them is regularity rather than difficulty.

**The figure eight** (`common.reference.figure_eight`) is what the budget table
is quoted against:

    x = a sin(w t),   y = (a/2) sin(2 w t),   z = h + c sin(w t / 2)

with `a = 1 m`, `c = 0.4 m`, `h = 1 m`, `w = 2 pi / 4.0`. Every frequency is a
rational multiple of every other, so the curve closes after one period and every
lap is identical. That makes a steady-state error meaningful, and it also lets a
controller's error lock into a pattern tied to the lap.

**The quasi-periodic path** (`common.reference.quasi_periodic`) puts the axes out
of phase forever:

    x = a sin(w t)
    y = 0.6 a sin(phi w t)
    z = h + c sin(w t / phi)

with `phi = 1.6180339887...`, the golden ratio, `a = 1 m`, `c = 0.35 m`,
`h = 1.2 m`. Because `phi` is irrational the axes never come back into phase
and **the path never repeats**: the horizontal projection slowly fills a
2.0 x 1.2 m box, and the altitude drifts on a third, slower cycle between 0.85
and 1.55 m. It is analytic, so the velocity handed to the controller is the
exact derivative of the position, which matters because the stage cost
penalizes linear velocity.

At 4.0 s the figure eight asks for 31 degrees of tilt at its peaks and the
quasi-periodic path for 23, so the bound is active on the first and slack on
the second; `run_figures.py` draws the arms on the quasi-periodic path by
default and takes `--reference figure8` and `--period` for the rest.

## Why the optimization phase had room

In the ADMM study both agents reported the same ceiling: 91% and 88% of runtime
inside `fom_lin_solve_t1` in `fom_linsolve.c`, a file regenerated before every
verification and therefore not theirs to edit. Here `src/fom_linsolve.c` is one
line long and declares no functions at all. The extracted iteration reports

```
extracted iteration:  2 steps, state dimension 400
parameters:           p0 (40), p1 (120)
schedules:            10 tables of 512
coefficient blocks:   24, 4240 applied entries
factorization:        none, 0 stored entries
work per iteration:   4240 multiply-adds
```

so every one of those 4240 multiply-adds is in `src/fom_solver.c`, which the
agent may rewrite. The count was 2680 before the tilt bound; the extra 1560 are
the two 20 x 40 blocks the bound adds, applied once each way, and they are the
only part of the iteration that is not a copy or a sign flip apart from the
condensed Hessian.

What the agent found there is in `out/codegen/optimized/OPTIMIZATIONS.md`. Of
the products the generic form applies, most are identity or selection blocks
and several are structurally zero; it dropped the zeros, turned the identities
into block copies and scaled sums, and merged the two dual terms that act on
the same rows of the state into one pass over a combined source. The two blocks
with real numbers in them it took apart: the condensed Hessian has a two-level
`[[a, b], [b, a]]` symmetry, so it applied a Hadamard butterfly and four
symmetric 10 x 10 products in place of one 40 x 40; and the 40 x 40 coupling
block the tilt rows introduce is, up to sign, transpose, and a Kronecker
factor, a single upper-triangular Toeplitz matrix with a ten-value kernel, so
each direction became two ten-point convolutions. Its final profile splits the
solve 49% / 26% / 24% between the primal pass, the state update, and the dual
pass, with nothing dominant left.

Two things differ from the earlier, unbounded run, and both are worth saying
plainly. The agent copied the numbers it needed into `fom_solver.c` as
constants rather than recovering them from `fom_ops.c` at setup, so
`fom_data.c` is no longer the only source of numbers; and the butterflies
reassociate, so the optimized and emitted solvers are not bit-identical. Both
solvers verify against the reference oracle to a worst absolute error of
1.3e-7, and against the NumPy controller at realistic parameters they agree to
1.5e-14 and 1.4e-14 respectively, 7.3e-14 and 4.9e-14 after two hundred
closed-loop iterations. The speedup is 19.9x per iteration and 20.4x per solve.

## Choosing the restart period

Two knobs have to be fixed before anything can be generated, and neither can be
read off the theory. Theorem 1 bounds the restart period K below by a maximum of
five terms, four of which involve the sharpness constant of the KKT system and
are not computable; the fifth, `||Q||/||A||`, is 2900 here, and the paper says
plainly that it "comes from the change of norm, and can be diminished by
rescaling the constraints and/or the objective" — a gauge artifact, not a
requirement. The primal weight is a practical device that PDQP.jl adapts online,
which a fixed table cannot do.

So `tuning.py` sweeps both, against how the closed loop flies at a fixed
iteration count rather than how fast one solve converges. Those disagree: a
controller always stops early, and a long restart period truncated mid-cycle
can be worse than a short one, so a configuration is scored on its geometric
mean error over the iteration counts a budget can land on *and* on whether that
error ever grows as the count grows. The sweep averages over the figure eight,
the quasi-periodic path, and a random B-spline, all at the 4.0 s period, flown
from the reference, on the tilt-bounded program:

| K | w = 1 | w = 2 | w = 4 |
|---:|---|---|---|
| 16 | 0.046 m | 0.080 m | 0.138 m |
| 32 | 0.034 m | 0.050 m, 1 regression (1.09x) | 0.084 m, 1 (1.19x) |
| **64** | **0.028 m** | 0.040 m | 0.059 m |
| 128 | 0.027 m, 2 regressions (1.35x) | 0.036 m | 0.050 m |

Typical error is the geometric mean over 16 iteration counts from 2 to 1024;
configurations not marked never track worse at a larger count than at a
smaller one. **K = 64, w = 1.** K = 128 is more accurate by four percent and
tracks worse at two budgets than at the one below them, by up to 1.35x, which
is the failure a controller cannot have; K = 64 never regresses. The primal
weight is one at every restart period, the floor of the range Lemma 1 allows:
the tilt rows raise `||G||` from `sqrt(2)` to nine, which already shrinks the
primal step against the dual one, and a larger weight slows the primal further
for no gain in how the bound is kept.

The crosses in the sweep are the random B-spline reference, which every
configuration loses at two and four iterations per solve and the heavier
weights at eight; the figure eight and the quasi-periodic path survive one
iteration per solve at a metre of error. Started on the reference at up to
2.3 m/s, a solve that has barely moved off the origin commands thrusts that
follow a gentle path badly and a sharp one not at all.

## How the restart survives code generation

The recording language has no restart. There is no nested loop and no branch:
extraction reads a straight line of steps and rolls it into one repeating body.
Unrolling the K inner steps is not available either — the term count grows like
K squared and `extract_unrolled` rejects it.

What makes the method expressible is that its coefficients are *tables*. The
inner index resets, so every coefficient is periodic in the global iteration
counter, and a schedule is any function of that counter. One recorded iteration
is one *inner* step, and the restart is a coefficient that is nonzero on every
K-th iteration:

    x_next = (1 - r) xhat + r xbar,    r = 1 when k = K-1, else 0

**K never enters the size of the emitted code.** It only sets numbers in the
tables, which is why a restart period of 32 costs exactly what a period of 2
would.

The table *length* is a real ceiling, though, in a way ADMM's `max_iter` is not:
past the end the emitted accessor clamps and the coefficients freeze, so
`common.study` refuses to fly a scheduled solver beyond it. Generation happened
at 512 because the generated benchmark runs five thousand solves at that length
and the optimization phase re-runs it after every change. Extending it
afterwards is free and needs no agent: the solver reads `FOM_MAX_ITER` and the
`fom_sched_*` accessors rather than any literal, so re-scaffolding regenerates
the tables and headers and leaves `src/fom_solver.c` untouched. Both solvers were
extended from 512 to 2048 that way and still verify, byte-identical.

Two properties of rAPDHG make the scheduled path apply where it would not for
ADMM: there is no factorization, so the check that rejects a scheduled system
matrix cannot fire; and the projection is a `relu`, which carries no scalar
constants, so the check that rejects a scheduled activation threshold cannot
fire either.

Two details in `recording.py` are load bearing. The gradient step is wrapped in a
saturated-linear step with bounds so wide they never bind, because extraction
folds every non-leaf expression into each place it is used and `xhat` is used by
three state slots — without it the one expensive product would be emitted three
times. And the right-hand side is declared a parameter, because the dual step
needs `tau_k * h` and a scheduled *constant offset* is rejected while a
scheduled coefficient on a parameter is not. With the tilt bound it is a
genuine parameter in any case: the thrust rows of `h` never move, the tilt rows
move with the measured state, and the generated solver takes both `q` and `h`
at every sampling instant.

## Checks

`python -m condensed.check_against_fomtools`, at K = 64, w = 1:

| check | result |
|---|---|
| structural `G` products vs the stored matrix | 1.776e-15 |
| flat recurrence vs Algorithm 1, across restarts | 2.762e-15 |
| extracted IR vs NumPy, 100 iterations | 3.587e-12 |
| converged rAPDHG vs Clarabel, relative objective | +0.000e+00 |
| relative KKT error at 100,000 iterations | 2.748e-15 |
| tilt slack of the Clarabel solution | +2.186e-02 rad |
| widest iterate seen vs the saturation bound | 0.188 vs 1e+06 |

The box rows of `G` are still applied structurally, as a copy and a sign flip;
the tilt rows are a real product, so the first line is round-off rather than
zero. Doing otherwise would charge the NumPy arm 4800 multiply-adds per product
for something code generation emits sparse and does in 1680, which would have
inflated the speedup.

### One number that is not what it looks like

`make verify` runs at `tol_verify = 1e-4` here, raised from the generator's 1e-6
default. That is a mis-scaled tolerance corrected, not a check relaxed.

The tolerance is **absolute**, and verification draws parameters uniformly from
the parameter boxes — the extremes each coordinate reaches over a whole flight.
A uniform draw from a forty-dimensional box sits near a corner, so those costs
are far larger than any single instant produces, and the iterate grows with
them, where a real flight's stays below 0.2 (the "widest iterate" line in the
checks table). Measured against the reference oracle the optimized solver's
worst absolute error over those draws is 1.3e-7, so 1e-4 leaves close to three
orders of margin.

The check that carries the experiment is unaffected and stays at 1e-6:
`check_equivalence` compares against NumPy at the representative mid-flight
parameters, where iterates are of order one, and both solvers pass it by eight
orders of magnitude. The ADMM study never met this because it generates at 20
iterations, so its states never grow.
