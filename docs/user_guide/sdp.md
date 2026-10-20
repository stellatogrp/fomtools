# Semidefinite relaxation (experimental)

!!! warning "Experimental"
    The SDP backend (`method="sdp"`) is experimental. Its bounds depend on an
    interior-point solver reaching an accurate optimum of a lifted semidefinite
    program, which does not always happen. The tutorials use the exact MIP backend
    and the simulator only.

## What it computes

With `method="sdp"`, FOMTools relaxes the verification problem to a semidefinite
program following
[Verification of First-Order Methods for Parametric Quadratic Optimization](https://arxiv.org/abs/2403.03331).
When the solver terminates at an optimum, the dual objective is an upper bound on the
worst-case value of the verification problem, usually looser than the exact value from
`method="mip"`. The algorithm is written exactly as for the other backends; only the
`Verifier` changes:

```python
import fomtools as ft

vp = ft.Verifier(method="sdp", solver="clarabel", obbt_solver=None)
```

Use `solver="cvxpy"` to solve the relaxation through CVXPY instead, or
`solver="cuclarabel"` on a GPU (this needs Julia, Clarabel.jl with CUDA, and
`juliacall`; see [Installation](../installation.md)). ReLU steps are supported directly;
soft-thresholding and saturated linear steps are represented through equivalent ReLU
decompositions.

## Tightening the relaxation

Optional cuts tighten the relaxation at additional computational cost. `rlt_cuts` adds
reformulation-linearization cuts between paired variables, `triangle_cuts` adds lifted
triangle cuts for ReLU outputs with mixed-sign bounds, and `full_psd_cone` merges the
moment matrices into one full PSD cone:

```python
vp = ft.Verifier(
    method="sdp",
    solver="clarabel",
    rlt_cuts=True,
    triangle_cuts=True,
    full_psd_cone=True,
)
```

## Keeping iterates as separate variables

An affine update such as `z[k] = M @ z[k-1] + c` produces a collapsed linear expression
that references existing variables. This suits the MIP and simulator backends. In the SDP
backend, collapsing couples many variables in one dense moment matrix; keeping each iterate
as its own variable gives a sparser matrix that splits into smaller PSD cones and is
usually faster to solve. The two choices give different relaxations.
`vp.iterate_trajectory(K + 1, collapse_expr="auto")` collapses on MIP and the simulator
and keeps separate variables on SDP; pass `collapse_expr=True` or `False` to force either.

## Known limitations

- **Solver termination.** With Clarabel, the bound is returned when it reports `Solved`, and
  with a warning when it reports `AlmostSolved`; the bound then holds only to Clarabel's
  reduced tolerances. Any other status (for example `InsufficientProgress` or
  `NumericalError`) raises `RuntimeError`, and no bound is returned. The CVXPY path accepts
  `optimal`, accepts `optimal_inaccurate` with a warning, and raises otherwise.
- **Conditioning.** The lifted problems are often poorly conditioned. On the ADMM iteration
  of the [portfolio tutorial](../tutorials/verification.md), for example, Clarabel reports
  `AlmostSolved` at every horizon and stops with `InsufficientProgress` at $K = 2$.
  Changing the solver settings moves the failure without removing it.
- **Scale.** The moment matrices grow with the dimension and the horizon, so solve times
  increase quickly; on the optimal control example the relaxation can take a long time to
  solve even at small $K$.
- **Numerical bounds.** Like all FOMTools bounds, these are numerical results subject to
  solver tolerances, not exact-arithmetic proofs.

For guarantees on a family of problems, prefer the exact MIP backend and use the
simulator for larger instances; see [Choosing a backend](index.md#choosing-a-backend).
