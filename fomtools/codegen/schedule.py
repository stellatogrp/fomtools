"""The iteration-varying coefficient *type*: a scalar tabulated over a loop.

This module holds only the data structure and its input coercion. The extraction
pass that recovers such coefficients from a recording and lowers them into the IR
is `fomtools.codegen.extract_scheduled`.

Some first-order methods use a coefficient that changes with the iteration
index: an accelerated method's momentum, a decaying step size, an anchor weight.
The verifier handles these for free, because it records an *unrolled* tape and
every iteration simply gets its own number. Rolling that tape back into a loop is
where the dependence on ``k`` would otherwise be lost -- declaring it as a
schedule is what preserves it.

A `Schedule` carries such a coefficient as a table of values, one per
iteration, computed here in Python at generation time. The emitted C reads the
table at the loop counter, so the iteration stays division-free and needs no
``libm`` call and no extra workspace state:

```c
static const fom_float FOM_SCHED_gamma[FOM_MAX_ITER] = { 0.0, 0.333, ... };

for (it = 0; it < work->settings.max_iter; ++it) {
    fom_float g = FOM_SCHED_gamma[it];
    ...
}
```

Indexing convention
-------------------
The index is the **0-based** loop counter, matching the emitted ``it``. Textbook
formulas are usually written in 1-based ``k``, so shifting them is the caller's
job: the coefficient belongs to the method being generated, not to the generator.
Nothing here knows what any particular method's coefficient is.

Only needed to get a *compact* solver for a method whose coefficients vary. The
same method can be recorded as an ordinary loop and generated as written, with no
schedule declared at all -- see `fomtools.codegen.extract_unrolled`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "Schedule",
    "as_schedule",
]


@dataclass(frozen=True)
class Schedule:
    """A scalar coefficient tabulated over the iterations of a generated loop.

    Attributes
    ----------
    name:
        Stable identifier used by the IR and the emitted code (a valid C
        identifier; it becomes ``FOM_SCHED_<name>``).
    values:
        One value per iteration, ``values[k]`` for the 0-based iteration ``k``.
        Must be at least as long as the generated loop's ``max_iters``.
    """

    name: str
    values: np.ndarray

    def __post_init__(self):
        arr = np.asarray(self.values, dtype=float).reshape(-1)
        if arr.size == 0:
            raise ValueError(f"schedule {self.name!r} is empty")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"schedule {self.name!r} contains non-finite values")
        object.__setattr__(self, "values", arr)

    def __len__(self) -> int:
        return int(self.values.size)


def as_schedule(name: str, spec, n_iters: int) -> Schedule:
    """Coerce a user-supplied schedule into a `Schedule` of length ``n_iters``.

    Parameters
    ----------
    name:
        Identifier for the schedule.
    spec:
        A `Schedule`, a callable mapping the 0-based iteration index to a
        float, or a sequence of at least ``n_iters`` values.
    n_iters:
        Number of iterations the generated loop can run, i.e.
        ``config.termination.max_iters``.

    Raises
    ------
    ValueError
        If a sequence or `Schedule` is shorter than ``n_iters``. The
        generated loop can run that far, so a shorter table would be read out of
        bounds; supply a callable instead if the horizon is not known up front.
    """
    if isinstance(spec, Schedule):
        if len(spec) < n_iters:
            raise ValueError(
                f"schedule {name!r} has {len(spec)} entries but the generated loop "
                f"can run {n_iters} iterations; supply a callable or a longer table"
            )
        return Schedule(name=name, values=spec.values[:n_iters])
    if callable(spec):
        return Schedule(name=name, values=np.array([float(spec(k)) for k in range(n_iters)]))
    if isinstance(spec, (Sequence, np.ndarray)):
        arr = np.asarray(spec, dtype=float).reshape(-1)
        if arr.size < n_iters:
            raise ValueError(
                f"schedule {name!r} has {arr.size} entries but the generated loop "
                f"can run {n_iters} iterations; supply a callable or a longer table"
            )
        return Schedule(name=name, values=arr[:n_iters])
    raise TypeError(
        f"schedule {name!r} must be a Schedule, a callable, or a sequence of floats, "
        f"got {type(spec).__name__}"
    )
