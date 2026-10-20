# Proxable functions

Proxable functions describe the function class an algorithm is verified over,
each supplying the proximal operator and the representation needed by the
verification problem. Use `vp.prox(function, arg, gamma=1.0)`; support varies by
backend, and unsupported operations raise explicitly. They all derive from the
abstract base class below.

::: fomtools.proxable_functions.ProxableFunction
    options:
      show_root_heading: true
      show_root_toc_entry: true

::: fomtools.proxable_functions
    options:
      show_root_heading: false
      show_root_toc_entry: false
      filters:
        - "!^ProxableFunction$"

## Sets and projections

Use `vp.project(set_instance, arg)`.

::: fomtools.sets.Set
::: fomtools.sets.Polyhedron
::: fomtools.sets.HalfSpace
::: fomtools.sets.Simplex
::: fomtools.sets.Box
::: fomtools.sets.NonNegative
::: fomtools.sets.NonPositive
::: fomtools.sets.Point
::: fomtools.sets.L1Ball
::: fomtools.sets.LinfBall
::: fomtools.sets.AffineSubspace
