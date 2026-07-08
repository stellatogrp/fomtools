"""Tests for the Set hierarchy and the vp.project(...) API.

``project`` is a thin wrapper over ``prox`` of a set's indicator, so these tests
cover: (1) the numeric Set API (project/contains) and the indicator specialization,
(2) the inheritance tree, (3) MIP-exact projection at a fixed arg (solver lowering vs
the numeric formula — an independent cross-check), (4) the class shorthand, (5) input
rejection, (6) SDP behavior, and (7) the simulator support split.
"""

import numpy as np
import pytest

from fomtools.linearexpression import LinearExpression
from fomtools.proxable_functions import NormL1
from fomtools.sets import (
    AffineSubspace,
    Box,
    HalfSpace,
    L1Ball,
    LinfBall,
    NonNegative,
    NonPositive,
    Point,
    Polyhedron,
    Set,
    Simplex,
)
from fomtools.verifier import Verifier
from tests.conftest import PROX_ATOL


def _make_verifier(method: str, obbt: str | None = None) -> Verifier:
    if method == "mip":
        return Verifier(method="mip", solver="gurobi", obbt_solver=obbt, params={"OutputFlag": 0})
    if method == "sdp":
        return Verifier(method="sdp", solver="clarabel", obbt_solver=obbt)
    if method == "simulator":
        return Verifier(method="simulator", obbt_solver=None)
    raise ValueError(f"unknown method: {method}")


# Each case: (id, set factory, fixed input x0, simulator-supported?, indicator class name).
# x0 is chosen to lie OUTSIDE the set so the projection is non-trivial (except Point).
_BOX_A = np.vstack([np.eye(3), -np.eye(3)])
_BOX_B = np.concatenate([np.ones(3), np.ones(3)])  # box [-1, 1]^3 as A x <= b

_CASES = [
    ("nonnegative", lambda: NonNegative(), np.array([-1.0, 2.0, -3.0]), True, "IndNonNegative"),
    ("nonpositive", lambda: NonPositive(), np.array([-1.0, 2.0, -3.0]), True, "IndNonPositive"),
    ("box", lambda: Box(-1.0, 1.0), np.array([2.0, -3.0, 0.5]), True, "IndBox"),
    (
        "point",
        lambda: Point(np.array([1.0, 2.0, 3.0])),
        np.array([0.0, 0.0, 0.0]),
        True,
        "IndPoint",
    ),
    ("linfball", lambda: LinfBall(1.0), np.array([2.0, -3.0, 0.5]), True, "IndBallLinf"),
    (
        "halfspace",
        lambda: HalfSpace(np.array([1.0, 1.0, 1.0]), 0.5),
        np.array([1.0, 1.0, 1.0]),
        True,
        "IndHalfSpace",
    ),
    (
        "polyhedron",
        lambda: Polyhedron(_BOX_A, _BOX_B, dual_ub=10.0),
        np.array([2.0, -3.0, 0.5]),
        True,  # per-sample cvxpy projection on the simulator
        "IndPolyhedral",
    ),
    (
        "affinesubspace",
        lambda: AffineSubspace(np.array([[1.0, 1.0, 1.0]]), np.array([1.0])),
        np.array([0.0, 0.0, 0.0]),
        True,  # IndAffine lowers to a batched Affine block-KKT solve on the simulator
        "IndAffine",
    ),
    ("simplex", lambda: Simplex(1.0), np.array([0.5, 0.2, 0.9]), True, "IndSimplex"),
    ("l1ball", lambda: L1Ball(1.0), np.array([0.8, 0.7, -0.6]), True, "IndBallL1"),
]
_IDS = [c[0] for c in _CASES]


# --- numeric Set API + indicator specialization ----------------------------


@pytest.mark.parametrize("name, make, x0, _sim, ind_name", _CASES, ids=_IDS)
def test_indicator_specialization_and_numeric(name, make, x0, _sim, ind_name):
    """Each set returns its most-specific indicator, and project/contains reuse it."""
    s = make()
    assert isinstance(s, Set)
    assert type(s.indicator()).__name__ == ind_name

    proj = s.project(x0)
    # project == indicator.prox (the contract); and the projection lands in the set.
    np.testing.assert_allclose(proj, s.indicator().prox(x0), atol=PROX_ATOL)
    # `contains` re-projects (idempotence test), so for cvxpy-solved sets (Simplex,
    # L1Ball, Polyhedron) it inherits the solver's ~PROX_ATOL precision, not machine eps.
    assert s.contains(proj, tol=PROX_ATOL)


def test_numeric_closed_forms():
    """Independent closed forms for the clip-style sets."""
    x = np.array([2.0, -3.0, 0.5])
    np.testing.assert_allclose(NonNegative().project(x), np.maximum(x, 0.0), atol=PROX_ATOL)
    np.testing.assert_allclose(NonPositive().project(x), np.minimum(x, 0.0), atol=PROX_ATOL)
    np.testing.assert_allclose(Box(-1.0, 1.0).project(x), np.clip(x, -1.0, 1.0), atol=PROX_ATOL)
    np.testing.assert_allclose(LinfBall(1.0).project(x), np.clip(x, -1.0, 1.0), atol=PROX_ATOL)
    np.testing.assert_allclose(
        Point(np.array([1.0, 2.0, 3.0])).project(x), np.array([1.0, 2.0, 3.0]), atol=PROX_ATOL
    )


def test_contains_inside_and_outside():
    assert NonNegative().contains(np.array([0.0, 1.0, 2.0]))
    assert not NonNegative().contains(np.array([-1.0, 1.0, 2.0]))
    assert Box(-1.0, 1.0).contains(np.array([0.5, -1.0, 1.0]))
    assert not Box(-1.0, 1.0).contains(np.array([0.5, -1.0, 1.5]))


# --- inheritance tree -------------------------------------------------------


def test_inheritance_tree():
    """Nominal inheritance mirrors the mathematics."""
    assert issubclass(Polyhedron, Set)
    for cls in (HalfSpace, Simplex, Box):
        assert issubclass(cls, Polyhedron)
    for cls in (NonNegative, NonPositive, Point):
        assert issubclass(cls, Box)
    assert issubclass(LinfBall, Box)
    # equality / lifted sets sit directly under Set
    assert issubclass(AffineSubspace, Set) and not issubclass(AffineSubspace, Polyhedron)
    assert issubclass(L1Ball, Set) and not issubclass(L1Ball, Polyhedron)
    # instance checks
    assert isinstance(NonNegative(), Polyhedron)
    assert isinstance(LinfBall(1.0), Box)


def test_subclasses_populate_box_attributes():
    """IS-A implies HAS-A: Box subclasses set the canonical (l, u) and parent attrs."""
    assert (NonNegative().l, NonNegative().u) == (0.0, np.inf)
    assert (NonPositive().l, NonPositive().u) == (-np.inf, 0.0)
    p = Point(np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(p.l, p.p) and np.array_equal(p.u, p.p)
    b = LinfBall(2.0)
    assert (b.l, b.u) == (-2.0, 2.0)
    # parent (Polyhedron) attributes exist too
    for s in (NonNegative(), NonPositive(), Point(0.0), LinfBall(1.0)):
        assert s.A is None and s.b is None and s.dual_ub is None


def test_repr_echoes_constructor_args():
    """__repr__ reads like the constructor (params no longer dropped)."""
    assert repr(Box(-1.0, 1.0)) == "Box(l=-1.0, u=1.0)"
    assert repr(NonNegative()) == "NonNegative()"
    assert repr(NonPositive()) == "NonPositive()"
    assert repr(LinfBall(2.0)) == "LinfBall(r=2.0)"
    assert repr(L1Ball(0.5, dual_ub=10.0)) == "L1Ball(r=0.5, dual_ub=10.0)"
    assert repr(Simplex(1.0)) == "Simplex(a=1.0)"
    assert repr(Point(0.5)) == "Point(p=0.5)"
    # array-valued args appear by value when small
    assert repr(HalfSpace(np.array([1.0, 2.0]), 0.5)) == "HalfSpace(a=[1., 2.], b=0.5)"
    # large arrays summarize by shape, not a wall of numbers
    big = AffineSubspace(np.ones((1, 50)), np.array([1.0]))
    assert "shape=(1, 50)" in repr(big)


def test_contains_is_scale_aware():
    """rtol gives a relative slack so large-magnitude points aren't held to a tiny
    absolute bound; rtol=0.0 recovers the pure-absolute behavior."""
    box = Box(-1e9, 1e9)
    pt = np.array([1e9 + 50.0])  # outside by 50, but ~5e-8 in relative terms
    assert box.contains(pt)  # default rtol=1e-5 -> within tolerance at this scale
    assert not box.contains(pt, rtol=0.0)  # pure absolute -> 50 > atol


def test_halfspace_uses_polyhedron_attribute_meaning():
    """HalfSpace stores A/b with the inherited Polyhedron meaning (1-row, RHS vector),
    not a scalar — resolving the self.b semantic collision. The normal is on self.a."""
    a = np.array([1.0, 2.0, -1.0])
    hs = HalfSpace(a, 0.5)
    np.testing.assert_array_equal(hs.A, a.reshape(1, -1))  # (1, n) matrix
    np.testing.assert_array_equal(hs.b, np.array([0.5]))  # length-1 RHS vector
    np.testing.assert_array_equal(hs.a, a)  # scalar-inequality normal
    assert hs.dual_ub is None
    # indicator still gets the scalar offset and projects correctly
    x = np.array([1.0, 1.0, 1.0])
    expected = x - max(0.0, (a @ x - 0.5)) / (a @ a) * a
    np.testing.assert_allclose(hs.project(x), expected, atol=PROX_ATOL)


# --- MIP-exact projection (solver lowering vs numeric formula) --------------


@pytest.mark.parametrize("name, make, x0, _sim, _ind", _CASES, ids=_IDS)
def test_mip_projection_matches_numeric(name, make, x0, _sim, _ind):
    """At a fixed arg, the MIP encoding of project reproduces the numeric projection."""
    v = _make_verifier("mip", obbt="gurobi")
    x = v.add_param(len(x0), lb=x0, ub=x0)  # fixed arg
    u = v.project(make(), x)
    assert isinstance(u, LinearExpression)
    assert u.ndim == len(x0)
    v.set_objective_norm([u], np.inf)
    v.solve()
    np.testing.assert_allclose(v.get_solution_value(u), make().project(x0), atol=PROX_ATOL)


# --- class shorthand --------------------------------------------------------


def test_class_shorthand_matches_instance():
    """project(NonNegative, arg) auto-instantiates and matches project(NonNegative(), arg)."""
    x0 = np.array([-1.0, 2.0, -3.0])

    v_cls = _make_verifier("mip", obbt="gurobi")
    u_cls = v_cls.project(NonNegative, v_cls.add_param(3, lb=x0, ub=x0))
    v_cls.set_objective_norm([u_cls], np.inf)
    v_cls.solve()

    v_inst = _make_verifier("mip", obbt="gurobi")
    u_inst = v_inst.project(NonNegative(), v_inst.add_param(3, lb=x0, ub=x0))
    v_inst.set_objective_norm([u_inst], np.inf)
    v_inst.solve()

    np.testing.assert_allclose(
        v_cls.get_solution_value(u_cls), v_inst.get_solution_value(u_inst), atol=PROX_ATOL
    )
    np.testing.assert_allclose(
        v_cls.get_solution_value(u_cls), NonNegative().project(x0), atol=PROX_ATOL
    )


# --- input rejection --------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [NormL1(0.5), object(), "nonnegative"],
    ids=["penalty", "object", "string"],
)
def test_project_rejects_non_set(bad):
    """project only accepts Sets; penalties/losses (and junk) raise TypeError."""
    v = _make_verifier("mip", obbt="gurobi")
    arg = v.add_param(3, lb=-1.0, ub=1.0)
    with pytest.raises(TypeError, match="Set"):
        v.project(bad, arg)


def test_project_parameterized_bare_class_helpful_error():
    """Passing a parameterized Set *class* (not instance) gives a project-aware error,
    not a bare 'missing required arguments' TypeError from the constructor."""
    v = _make_verifier("mip", obbt="gurobi")
    arg = v.add_param(3, lb=-1.0, ub=1.0)
    with pytest.raises(TypeError, match="parameterless|instantiate it explicitly"):
        v.project(Box, arg)  # Box needs l, u


# --- SDP backend ------------------------------------------------------------


@pytest.mark.parametrize(
    "make, x0",
    [
        (lambda: NonNegative(), np.array([-1.0, 2.0, -3.0])),  # step-based
        (lambda: Polyhedron(_BOX_A, _BOX_B, dual_ub=10.0), np.array([2.0, -3.0, 0.5])),  # KKT-based
    ],
    ids=["nonnegative", "polyhedron"],
)
def test_sdp_projection_tight_at_fixed_point(make, x0):
    """At a fixed arg, max ||u||^2 over the SDP relaxation equals ||proj(x0)||^2."""
    v = _make_verifier("sdp")
    x = v.add_param(len(x0), lb=x0, ub=x0)
    u = v.project(make(), x)
    v.set_objective_norm(u, 2)
    obj = v.solve()
    proj = make().project(x0)
    np.testing.assert_allclose(obj, float(proj @ proj), atol=1e-3)


# --- simulator support split ------------------------------------------------


@pytest.mark.parametrize("name, make, x0, sim_supported, _ind", _CASES, ids=_IDS)
def test_simulator_support_matches_prox(name, make, x0, sim_supported, _ind):
    """Every Set projection now runs on the simulator: clip/step-based ones as
    primitives, the affine KKT solve as a batched linear solve, and the
    simplex/ℓ1-ball/polyhedral projections via batched / per-sample evaluators. The
    ``sim_supported`` flag is retained (all True) so a future deferral is caught."""
    v = _make_verifier("simulator")
    arg = v.add_param(len(x0), lb=x0, ub=x0)
    if sim_supported:
        out = v.project(make(), arg)
        assert isinstance(out, LinearExpression)
    else:
        with pytest.raises(NotImplementedError):
            v.project(make(), arg)
