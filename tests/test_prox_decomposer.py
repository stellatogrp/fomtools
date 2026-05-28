"""
Integration tests for the proxable-function decomposer layer.

Covers the contract between ``Verifier.prox`` and
``StepDecomposer.process_prox`` / ``_PROX_DECOMPOSERS``:

- cross-backend inheritance of base-class registrations,
- recursive interaction with the step decomposer registry, and
- the ``NotImplementedError`` path for proxes not registered on a
  given backend.

Numerical correctness of individual prox math (e.g. ``NormL1.prox``
returning the right soft-threshold output) is the concern of
``tests/test_prox_eval.py``, not this file.
"""

import numpy as np
import pytest

from fomtools.linearexpression import LinearExpression
from fomtools.proxable_functions import NormL1, ProxableFunction
from fomtools.steps import Relu, SoftThreshold
from fomtools.verifier import Verifier


def _make_verifier(method: str, obbt: str | None = None) -> Verifier:
    """Construct a Verifier for the given backend with sensible defaults."""
    if method == "mip":
        return Verifier(
            method="mip",
            solver="gurobi",
            obbt_handler=obbt,
            params={"OutputFlag": 0},
        )
    if method == "sdp":
        return Verifier(method="sdp", solver="clarabel", obbt_handler=obbt)
    if method == "simulator":
        return Verifier(method="simulator", obbt_handler=None)
    raise ValueError(f"unknown method: {method}")


@pytest.mark.parametrize("method", ["mip", "sdp", "simulator"])
def test_norml1_returns_linear_expression(method):
    """NormL1 prox is registered on the base class; every backend inherits."""
    v = _make_verifier(method)
    rhs = v.add_param(3, lb=-1.0, ub=1.0)

    out = v.prox(NormL1(lambd=0.3), rhs, gamma=0.5)

    assert isinstance(out, LinearExpression)
    assert out.ndim == 3


def test_norml1_emits_softthreshold_on_mip():
    """
    On MIP, NormL1 lowers to a single SoftThreshold primitive (no recursive
    decomposition — MIP encodes SoftThreshold natively).
    """
    v = _make_verifier("mip")
    rhs = v.add_param(3, lb=-1.0, ub=1.0)
    v.prox(NormL1(lambd=0.3), rhs, gamma=0.5)

    n_soft = sum(
        1 for s in v.canonicalizer.steps if isinstance(s, SoftThreshold)
    )
    assert n_soft == 1
    # SoftThreshold's lambd should be the scaled value lambd * gamma = 0.15
    soft_step = next(
        s for s in v.canonicalizer.steps if isinstance(s, SoftThreshold)
    )
    assert np.isclose(soft_step.lambd, 0.15)


def test_norml1_decomposes_to_two_relus_on_sdp():
    """
    On SDP, the SoftThreshold emitted by prox_norml1 recursively decomposes
    into two ReLUs via the existing _DECOMPOSERS registry — confirms that
    the two registry layers (prox + step) compose correctly.
    """
    v = _make_verifier("sdp")
    rhs = v.add_param(3, lb=-1.0, ub=1.0)
    v.prox(NormL1(lambd=0.3), rhs, gamma=0.5)

    # SDP canonicalizer tracks Relus it encoded
    assert len(v.canonicalizer._relu_steps) == 2
    assert all(isinstance(s, Relu) for s in v.canonicalizer._relu_steps)


def test_unsupported_prox_raises_on_every_backend():
    """
    A ProxableFunction subclass not registered in any _PROX_DECOMPOSERS dict
    raises NotImplementedError with a backend-named message on every backend.
    """

    class Unregistered(ProxableFunction):
        pass

    for method in ("mip", "sdp", "simulator"):
        v = _make_verifier(method)
        rhs = v.add_param(3, lb=-1.0, ub=1.0)

        with pytest.raises(NotImplementedError, match="Unregistered"):
            v.prox(Unregistered(), rhs)


def test_unsupported_prox_error_names_the_backend():
    """The error message identifies which decomposer rejected the prox."""

    class Unregistered(ProxableFunction):
        pass

    v = _make_verifier("mip")
    rhs = v.add_param(3, lb=-1.0, ub=1.0)

    with pytest.raises(NotImplementedError, match="MIPStepDecomposer"):
        v.prox(Unregistered(), rhs)
