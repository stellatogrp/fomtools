# Uncondensed formulation

Nothing is eliminated.  The states stay as variables and the dynamics stay as
constraints,

```
minimize    (1/2) zeta' P zeta + q(r)' zeta
subject to  l(x) <= M zeta <= u(x),
```

over `zeta = (xi_1, ..., xi_100, z_1, ..., z_10)`: 1240 variables and 1240
constraint rows at the same one second lookahead the condensed study uses, in
three groups — the initial state, the dynamics, and the input box.

Everything is banded.  `P` is diagonal, each dynamics row touches two
consecutive states and one input move, and the system matrix
`P + sigma I + M' R M` holds 15188 entries out of 1.5 million, 0.99% filled.
Its sparse factorization holds 26739, against the 769000 a dense factor of that
size would hold.  Every solve applies that factorization; nothing is refactorized
inside the loop.

Two consequences distinguish this formulation from the condensed one, beyond
size.

**The measured state enters through the bounds.**  `xi_1 = x` is an equality,
and an equality is a bound whose two sides coincide, so the first twelve rows of
`l` and `u` move every sampling instant while the rest stay fixed.  The
condensed formulation puts the measured state in the linear cost instead.  This
matters for code generation and not at all for the simulation; see the note at
the end.

**The equality rows need their own penalty.**  `rho_equality` defaults to 1000,
which is OSQP's own convention for equality constraints, and it is not optional
here.  At the same penalty as the inequality rows the iterate buys objective by
violating the dynamics:

| equality penalty | objective gap at 5000 iterations | worst dynamics violation |
|---:|---:|---:|
| 0.1 (same as the box) | -0.56 | 1.1e-1 |
| 1 | -0.33 | 2.5e-2 |
| 10 | -0.11 | 3.5e-3 |
| 100 (the default) | -6.0e-6 | 3.1e-7 |

A negative gap means the iterate scores *better* than the true optimum, which it
can only do by being infeasible.  At the low settings the controller is
optimizing over motions the quadcopter cannot perform, and it is still doing so
after five thousand iterations.  The condensed formulation has no equality rows
for any of this to apply to: its dynamics are substituted away and every row of
its constraint matrix is a thrust bound.

## Running it

```
python -m uncondensed.check_against_fomtools
python -m uncondensed.run_budget_sweep
```

Outputs land in `out/`, with the same names and meanings as in `condensed/`.

## Calibration

Eight second flights, 10 ms sampling, error in metres, steady excluding the
first two seconds.

| iterations | outcome | steady error | thrusts on the box |
|---:|---|---:|---:|
| 10 | flew | 2.05 m | 86% |
| 25 | flew | 2.04 m | 90% |
| 50 | flew | 2.27 m | 85% |
| 100 | flew | 1.53 m | 83% |
| 250 | flew | 0.30 m | 92% |
| 500 | flew | 0.12 m | 80% |
| 1000 | flew | 0.12 m | 66% |
| 2500 | flew | 0.12 m | 58% |

The converged error, 0.119 m, is the same as the condensed study reaches, which
is the strongest evidence available that the two programs describe one control
problem; `compare.py` checks it directly on the optimal first input.

The approach to it is different, though.  This controller does not crash until
the budget falls below about ten iterations, where the condensed one is already
gone at a hundred.  Starved, it degrades into a sluggish controller rather than
a wrong one: the large penalty on the equality rows keeps the predicted
trajectory roughly consistent with the dynamics even when the solve is nowhere
near converged, so the first input stays sane.  That robustness does not make it
the better choice at a fixed time budget — see the parent README — but it is a
real difference and not an artifact.

A NumPy iteration costs about 36.5 us here, six times the condensed one for
thirty-one times the variables.

## Checks

`check_against_fomtools.py` runs the same three executions the condensed study
does — the NumPy controller, `run_faithful` on the recorded steps, and
`ReferenceOracle` on the extracted representation — and they agree to about
3e-11.  It also confirms the converged iterate matches OSQP to 1e-12 relative on
the objective and satisfies the dynamics to 2e-15.

The recording fixes the constraint bounds at one representative state.  That is
enough for what the check establishes, since the iteration is the same map
whatever numbers sit in the bounds, and it is what the simulation does anyway:
bounds are supplied as numbers refreshed each instant, never as symbolic
parameters.

A *deployed* solver would need more.  Its bounds would have to be recomputed
from the measured state inside the generated loop, and `fomtools.codegen` cannot
express that today: the intermediate representation stores saturated-linear
bounds as names of constant arrays, and `extract.py:_satlin_arrays` raises
`NonStationaryFOMError` on a bound that is a `LinearExpression`.  Lifting that
restriction means letting `ActivationOp` hold a linear operator instead of a
matrix name and having the emitter assemble the bounds with the `fom_apply_*`
helpers it already generates.  The factorization is untouched by any of it,
since `P + sigma I + M' R M` does not depend on `l` or `u`.
