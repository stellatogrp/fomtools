"""Convex-envelope rows for piecewise-linear steps, shared by the MIP and OBBT canonicalizers.

Each helper returns a list of ``Constraint`` rows in ``output`` and ``arg`` and
handles infinite input bounds by writing the limit of the chord instead of a
NaN coefficient.
"""

import numpy as np


def relu_upper_envelope(output, arg, lb: float, ub: float) -> list:
    """Upper side of the ReLU triangle envelope for ``lb < 0 < ub``.

    With finite bounds this is the chord ``output <= ub/(ub-lb) * (arg - lb)``.
    An infinite bound has no chord, so the cut is replaced by its limit:
    ``output <= arg - lb`` when only ``ub`` is infinite, ``output <= ub`` when
    only ``lb`` is infinite, and nothing when both are. Writing the chord
    with an infinite bound would hand the solver NaN coefficients.
    """
    if np.isfinite(lb) and np.isfinite(ub):
        return [output <= ub / (ub - lb) * (arg - lb)]
    if np.isfinite(lb):
        return [output <= arg - lb]
    if np.isfinite(ub):
        return [output <= ub]
    return []


def softthreshold_envelope(
    output, arg, lb: float, ub: float, output_lb: float, output_ub: float, lambd: float
) -> list:
    """Envelope rows of the soft-threshold graph for an input range that crosses a kink.

    Covers the three mixed cases (two-sided ``lb < -lambd < lambd < ub``,
    right-or-middle and left-or-middle). The caller must handle the fixed
    regions (``lb >= lambd``, ``ub <= -lambd`` and ``[lb, ub]`` inside
    ``[-lambd, lambd]``) first; any other input raises ``RuntimeError``.

    With finite bounds these are the chord rows. An infinite bound has no
    chord, so the chord is replaced by its limit, which has slope 1: in the
    two-sided case that limit is ``arg +/- lambd``, already present, so the
    chord is dropped; one-sided it becomes ``output <= arg - lb`` or
    ``output >= arg - ub``. The choice uses only the input bounds: a finite
    output bound with an infinite input bound would give slope 0 and an
    invalid row.
    """
    if lb < -lambd and ub > lambd:
        rows = [output >= arg - lambd, output <= arg + lambd]
        if np.isfinite(ub):
            rows.append(output <= output_ub / (ub + lambd) * (arg + lambd))
        if np.isfinite(lb):
            rows.append(output >= output_lb / (lb - lambd) * (arg - lambd))
        return rows
    if -lambd <= lb <= lambd < ub:
        if np.isfinite(ub):
            chord = output <= output_ub / (ub - lb) * (arg - lb)
        else:
            chord = output <= arg - lb
        return [output >= 0, chord, output >= arg - lambd]
    if lambd >= ub >= -lambd >= lb:
        if np.isfinite(lb):
            chord = output >= output_lb / (lb - ub) * (arg - ub)
        else:
            chord = output >= arg - ub
        return [output <= 0, chord, output <= arg + lambd]
    raise RuntimeError("Unreachable code: soft-threshold input range does not cross a kink")
