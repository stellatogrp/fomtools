# rAPDHG on the condensed quadcopter program

Forty variables, eighty inequality rows, one second of lookahead at a 10 ms
sampling period. The same control problem the `osqp` study flies — the plant,
the cost, the reference, and the move blocking all come from
`../../common/condensed_qp.py`, shared between the two — solved by a method with
no linear system in it.

## Running it

```
python -m condensed.check_against_fomtools    # the controller is the recorded algorithm
python -m condensed.generate                  # both C solvers (FOMTOOLS_CODEGEN_AGENT=1)
python -m condensed.run_implementations       # the three arms at equal budgets
python -m condensed.run_figures               # the same three on a non-repeating path
python -m condensed.tuning                    # the K and primal-weight sweep
```

## The result

| implementation | us/iteration | spread | vs NumPy | vs emitted |
|---|---:|---:|---:|---:|
| NumPy | 10.02 | 1.8% | 1.0x | |
| emitted C (phase A) | 2.38 | 0.9% | **4.2x** | 1.0x |
| optimized C (phase B) | 0.133 | 0.7% | **75x** | **17.9x** |

Set against the ADMM study, where emission bought 2.1x and the optimization
phase 1.08x. The difference is not that this agent was cleverer. It is that
there was something left for it to optimize.

Flown for eight seconds against the figure eight, steady tracking error in
metres. No arm crashes at any budget.

| budget | NumPy | emitted C | optimized C |
|---:|---|---|---|
| 25 us | 2 it, 1.009 | 10 it, 0.541 | 186 it, **0.170** |
| 40 us | 3 it, 0.823 | 16 it, 0.383 | 297 it, **0.154** |
| 63 us | 6 it, 0.643 | 25 it, 0.253 | 469 it, **0.148** |
| 100 us | 9 it, 0.567 | 41 it, 0.263 | 744 it, **0.216** |
| 160 us | 15 it, 0.407 | 65 it, 0.218 | 1191 it, **0.114** |

The budgets are microseconds, two orders of magnitude below the ADMM study's
grid, and that is the finding rather than a setting: at 0.133 us an iteration
the optimized solver is converged inside a hundred microseconds, and a grid
running to a millisecond would show three flat lines. The floor is set by the
slowest arm — NumPy costs 10 us an iteration, so anything under about 25 us buys
it a single iteration and several budgets in a row report the same flight. The
sampling period is still 10 ms; what this measures is how small a slice of it
each implementation needs.

**The 100 us row is not a solver failure.** The optimized arm goes 0.148 at 469
iterations, 0.216 at 744, then 0.114 at 1191. Flying that same configuration on
four different trajectories isolates it:

| iterations | figure8 | spline0 | spline1 | spline2 |
|---:|---|---|---|---|
| 469 | 0.148 | 0.051 | 0.044 | 0.036 |
| 744 | **0.216** | 0.048 | 0.043 | 0.034 |
| 1191 | 0.114 | 0.049 | 0.042 | 0.033 |

Three of the four are monotone. The bump belongs to the figure eight, at one
particular solve accuracy — a closed-loop resonance with a path that repeats
every 2.25 s, not a property of the solver. That is what motivated the second
reference below.

## The trajectories

**The figure eight** (`common.reference.figure_eight`) is what every number
above is quoted against:

    x = a sin(w t),   y = (a/2) sin(2 w t),   z = h + c sin(w t / 2)

with `a = 1 m`, period `2.25 s`, `c = 0.4 m`, `h = 1 m`. Every frequency is a
rational multiple of every other, so the whole curve closes after one period and
every lap is identical. That is convenient — it makes a steady-state error
meaningful — and it is also why a controller's error can lock into a pattern
tied to the lap, which is what the 100 us row above is.

**The quasi-periodic path** (`common.reference.quasi_periodic`) is the same idea
with the axes deliberately put out of phase forever:

    x = a sin(w t)
    y = 0.6 a sin(phi w t)
    z = h + c sin(w t / phi)

with `phi = 1.6180339887...`, the golden ratio, and `a = 1 m`, period `2.25 s`,
`c = 0.35 m`, `h = 1.2 m`. Because `phi` is irrational — and the golden ratio is
the irrational hardest to approximate by a fraction — the three axes never come
back into phase and **the path never repeats**. The horizontal projection is a
Lissajous figure that slowly fills a 2.0 x 1.2 m box instead of retracing one
curve, and the altitude drifts on a third, slower incommensurate cycle between
0.85 and 1.55 m, clear of the ground.

It is analytic, so the velocity handed to the controller is the exact derivative
of the position rather than a finite difference of it. That matters here: the
stage cost penalizes linear velocity, so a reference whose velocity disagrees
with its position makes the cost fight the path.

The defaults keep it comparably demanding — peak speed 3.94 m/s against the
figure eight's 3.99, mean 2.69 against 2.76, the same 2 m of travel in `x` — so
the difference between the two is the regularity and not the difficulty.

`run_figures.py` draws the three arms on it. At the 63 us budget the arms
separate more widely than on the figure eight, and monotonically:

| | iterations | steady error |
|---|---:|---|
| NumPy | 6 | 0.691 m |
| emitted C | 26 | 0.207 m |
| optimized C | 469 | **0.077 m** |

Note the optimized arm reaches 0.077 m here against 0.148 m on the figure eight
at the same iteration count. It is not tracking a gentler path; it is tracking
one that gives the closed loop nothing periodic to resonate with.

## Why the optimization phase had room

In the ADMM study both agents reported the same ceiling: 91% and 88% of runtime
inside `fom_lin_solve_t1` in `fom_linsolve.c`, a file regenerated before every
verification and therefore not theirs to edit. Here `src/fom_linsolve.c` is one
line long and declares no functions at all. The extracted iteration reports

```
extracted iteration:  2 steps, state dimension 320
schedules:            10 tables of 2048
coefficient blocks:   24, 2680 applied entries
factorization:        none, 0 stored entries
work per iteration:   2680 multiply-adds
```

so every one of those 2680 multiply-adds is in `src/fom_solver.c`, which the
agent may rewrite.

What it found there is worth reading in its own header. Of the 22 `matrix @
source` products the generic form applies, **exactly one is dense** — the
condensed Hessian, reached through the `xmd` state slot. Every other one is
numerically a single signed identity block dropped into one slot of an otherwise
empty matrix. Applied through the CSR helpers those 21 products spend almost all
their time zeroing 320-long accumulators and walking row pointers over
structurally empty rows. Expanded into the block operations they actually are,
"22 matvecs over 3520 output rows collapse to ~1000 scalar operations plus one
40x40 dense product."

Two details make the result trustworthy rather than merely fast. The dense block
is not copied into the solver: it is recovered once in `fom_setup` by probing the
`fom_ops.c` helpers with unit vectors, so `fom_data.c` stays the only source of
numbers. And the rewrite is exact — term order and per-term rounding are
preserved, the dense block is stored column-major so the kernel vectorizes
"without reassociating a single floating-point addition", and only rewrites
IEEE-754 makes exact are used. The consequence is measurable: the optimized and
emitted solvers agree **bit for bit**, `0.000e+00` at every iteration count from
1 to 512.

That also tempers the headline. A 17.9x speedup is partly a statement about how
much dispatch overhead the phase-A translation carries on a problem whose blocks
are nearly all identities. In the ADMM study the same overhead existed and was
invisible, because it sat behind a linear solve that cost more than all of it.

## Choosing the restart period

Two knobs have to be fixed before anything can be generated, and neither can be
read off the theory. Theorem 1 bounds the restart period K below by a maximum of
five terms, four of which involve the sharpness constant of the KKT system and
are not computable; the fifth, `||Q||/||A||`, is 18250 here, and the paper says
plainly that it "comes from the change of norm, and can be diminished by
rescaling the constraints and/or the objective" — a gauge artifact, not a
requirement. The primal weight is a practical device that PDQP.jl adapts online,
which a fixed table cannot do.

So `tuning.py` sweeps both, against how the closed loop flies at a fixed
iteration count rather than how fast one solve converges. Those disagree, and
the disagreement is the interesting part: a controller always stops early, and a
long restart period truncated mid-cycle can be worse than a short one. **K = 32,
w = 2**, on three pieces of evidence.

*Too short and it stops converging.* At K = 8 the momentum coefficient never
exceeds 5. After 100,000 iterations its relative KKT error plateaus at 4.2e-04
and its objective is 1.2e-07 off, where K = 32 reaches 4.0e-09 and 1.4e-12.

*Too long and truncation becomes unpredictable.* At K = 64 with w = 1 the flight
crashes at 192 and 256 iterations and not at 512; at K = 128 and 256 it crashes
over wider bands. A controller that fails at some budgets and not at larger ones
is unusable however good its asymptotics.

*The primal weight is what makes K = 32 safe.* Over the swept iteration range the
worst regression as the budget grows is 4.4x at w = 1 and 1.4x at w = 2.

One caveat on the evidence. The full sweep reported here is on the figure eight,
and the 100 us row above shows how much one trajectory can decide: the spread
between the better configurations is a few percent, well inside what a single
reference settles by chance. `tuning.py` therefore averages over four
trajectories and rejects B-spline seeds whose path leaves the flyable region —
seed 0 at the default altitude descends to -0.00 m and crashes at every iteration
count including 1024, which is a property of the reference and not the solver.
That averaged sweep takes about 45 minutes and has not been run to completion
here; the K = 32 choice rests on the three findings above, each of which is
about a different failure mode rather than a few percent of tracking error.

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
times. And the right-hand side is declared a parameter over a degenerate box,
because the dual step needs `tau_k * h` and a scheduled *constant offset* is
rejected while a scheduled coefficient on a parameter is not.

## Checks

`python -m condensed.check_against_fomtools`:

| check | result |
|---|---|
| structural `G` products vs the stored matrix | 0.000e+00 |
| flat recurrence vs Algorithm 1, across restarts | 5.773e-15 |
| extracted IR vs NumPy, 100 iterations | 2.792e-11 |
| converged rAPDHG vs Clarabel, relative objective | +1.418e-12 |
| relative KKT error at 100,000 iterations | 4.020e-09 |
| widest iterate seen vs the saturation bound | 1.236 vs 1e+06 |

and from `run_implementations`, all three arms flown at a common 200 iterations
agree to **7.461e-14**.

`G = [I; -I]` is applied structurally rather than as a stored matrix, checked
exactly equal above. Doing otherwise would charge the NumPy arm 3200
multiply-adds per product for something code generation emits sparse and does in
80, which would have inflated the speedup.

### One number that is not what it looks like

`make verify` runs at `tol_verify = 1e-4` here, raised from the generator's 1e-6
default. That is a mis-scaled tolerance corrected, not a check relaxed.

The tolerance is **absolute**, and verification draws cost vectors uniformly from
the parameter box — the extremes each coordinate reaches over a whole flight. A
uniform draw from a forty-dimensional box sits near a corner, so those costs are
far larger than any single instant produces, and the iterate grows with them: it
reaches 2.6e+04 after 512 iterations where a real flight's is 1.2. Measured
against the reference oracle the emitted solver agrees to **4e-11 relative at
every sample**, with the error growing smoothly from 2.2e-13 at two iterations —
ordinary round-off. It is only the product of that relative error with an
iterate that large which crosses 1e-6. The measured worst case is 1.6e-6, so
1e-4 leaves close to two orders of margin.

The check that carries the experiment is unaffected and stays at 1e-6:
`check_equivalence` compares against NumPy at the representative mid-flight
parameters, where iterates are of order one, and both solvers pass it by eleven
orders of magnitude. The ADMM study never met this because it generates at 20
iterations, so its states never grow.
