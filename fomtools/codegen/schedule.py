"""Iteration-varying scalar coefficients for code generation.

Some first-order methods use a coefficient that changes with the iteration
index: FISTA's momentum :math:`\\gamma_k = (\\beta_{k-1} - 1)/\\beta_k`,
Nesterov's :math:`\\gamma_k = (k-1)/(k+2)`, a Halpern anchor weight
:math:`\\alpha_k = (k+1)/(k+2)`. The verifier handles these for free, because it
records an *unrolled* tape and every iteration simply gets its own number. Code
generation rolls the tape back into a loop, which is where the dependence on
``k`` would otherwise be lost.

A :class:`Schedule` carries such a coefficient as a table of values, one per
iteration, computed here in Python at generation time. The emitted C reads the
table at the loop counter, so the iteration stays division-free and needs no
``libm`` call and no extra workspace state:

.. code-block:: c

    static const fom_float FOM_SCHED_gamma[FOM_MAX_ITER] = { 0.0, 0.333, ... };

    for (it = 0; it < work->settings.max_iter; ++it) {
        fom_float g = FOM_SCHED_gamma[it];
        ...
    }

Indexing convention
-------------------
The index is the **0-based** loop counter, matching the emitted ``it``. A
schedule written from a textbook formula in 1-based ``k`` must be shifted; the
helpers below (:func:`nesterov_momentum`, :func:`fista_momentum`) already
account for this, so prefer them over hand-rolling the offset.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "Schedule",
    "as_schedule",
    "fista_momentum",
    "nesterov_momentum",
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

    def truncated(self, n: int) -> "Schedule":
        """Return this schedule cut to its first ``n`` entries."""
        if n > len(self):
            raise ValueError(f"schedule {self.name!r} has {len(self)} entries, need at least {n}")
        return Schedule(name=self.name, values=self.values[:n])


def as_schedule(name: str, spec, n_iters: int) -> Schedule:
    """Coerce a user-supplied schedule into a :class:`Schedule` of length ``n_iters``.

    Parameters
    ----------
    name:
        Identifier for the schedule.
    spec:
        A :class:`Schedule`, a callable mapping the 0-based iteration index to a
        float, or a sequence of at least ``n_iters`` values.
    n_iters:
        Number of iterations the generated loop can run, i.e.
        ``config.termination.max_iters``.

    Raises
    ------
    ValueError
        If a sequence or :class:`Schedule` is shorter than ``n_iters``. The
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


def nesterov_momentum() -> Callable[[int], float]:
    """Nesterov's momentum coefficient, in the 0-based index of the emitted loop.

    The textbook form is :math:`\\gamma_k = (k-1)/(k+2)` with ``k`` starting at
    1, so :math:`\\gamma_1 = 0`. Shifted to the 0-based loop counter this is
    ``k / (k + 3)``.
    """

    def gamma(k: int) -> float:
        return k / (k + 3.0)

    return gamma


def fista_momentum() -> Callable[[int], float]:
    """FISTA's momentum coefficient, in the 0-based index of the emitted loop.

    Generated from :math:`\\beta_0 = 1`,
    :math:`\\beta_{k+1} = (1 + \\sqrt{1 + 4\\beta_k^2})/2`, giving
    :math:`\\gamma_k = (\\beta_{k-1} - 1)/\\beta_k`. The first value is 0, which
    is exactly the degeneracy that makes a single-iteration recording of FISTA
    indistinguishable from ISTA, so the tabulated form is the safe one.

    The returned callable caches its own recurrence, so it must be called with
    non-decreasing indices — which :func:`as_schedule` does.
    """
    betas = [1.0]

    def gamma(k: int) -> float:
        while len(betas) <= k + 1:
            b = betas[-1]
            betas.append((1.0 + np.sqrt(1.0 + 4.0 * b * b)) / 2.0)
        return (betas[k] - 1.0) / betas[k + 1]

    return gamma
