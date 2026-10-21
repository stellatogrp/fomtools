"""Tests for ``tighten_residuals``: automatic triangle-inequality tightening after each solve.

After ``vp.maximize(ft.norm(a - b, np.inf))`` and ``vp.solve()``, the certified bound is
chained with the earlier ones and applied to ``a``, as a manual
``post_process(a, root, bounds)`` would.
"""

import numpy as np
import pytest

import fomtools as ft
from tests.conftest import TEST_EPS


def _fista(vp, K, solve=True, manual=False):
    """FISTA on a small lasso; returns ``(bounds, iterates)``."""
    m, n, lambd = 4, 6, 0.1
    D = np.random.default_rng(0).normal(size=(m, n))
    eta = 1 / np.linalg.norm(D, 2) ** 2
    x = vp.parameter(m, lb=0, ub=1)
    s = w = vp.iterate(n, lb=0, ub=0)
    t, bounds, iterates = 1.0, [], [s]
    for _ in range(K):
        s_prev, s = s, vp.soft_threshold(w - eta * D.T @ (D @ w - x), lambd=lambd * eta)
        t_next = 0.5 * (1 + np.sqrt(1 + 4 * t**2))
        w, t = s + (t - 1) / t_next * (s - s_prev), t_next
        iterates.append(s)
        if solve:
            vp.maximize(ft.norm(s - s_prev, np.inf))
            bounds.append(vp.solve())
            if manual:
                vp.post_process(s, iterates[0], bounds)
    return bounds, iterates


def test_subtraction_records_its_operands():
    vp = ft.Verifier()
    a, b = vp.parameter(2), vp.parameter(2)
    assert (a - b)._difference == (a, b)
    assert (a + b)._difference is None
    assert (a - np.ones(2))._difference is None


def test_worst_case_is_unchanged():
    on, _ = _fista(ft.Verifier(), K=4)
    off, _ = _fista(ft.Verifier(tighten_residuals=False), K=4)
    # Both are best bounds, equal up to HiGHS's default relative MIP gap (1e-4).
    np.testing.assert_allclose(on, off, rtol=2e-4)


def test_matches_manual_post_process():
    auto_vp, manual_vp = ft.Verifier(), ft.Verifier(tighten_residuals=False)
    _, auto = _fista(auto_vp, K=3)
    _, manual = _fista(manual_vp, K=3, manual=True)
    for a, m in zip(auto[1:], manual[1:]):
        np.testing.assert_allclose(a.lb, m.lb, atol=1e-6)
        np.testing.assert_allclose(a.ub, m.ub, atol=1e-6)


def test_chains_the_bounds_from_the_initial_iterate():
    vp = ft.Verifier()
    bounds, iterates = _fista(vp, K=3)
    s0 = iterates[0]
    for k, s in enumerate(iterates[1:], start=1):
        root, radius = vp._residual_chain[s.counter]
        assert root is s0
        assert radius == pytest.approx(sum(bounds[:k]), rel=1e-6)
        # s^0 = 0, so every entry of s^k lies in [-radius, radius].
        assert np.all(s.ub <= radius + TEST_EPS) and np.all(s.lb >= -radius - TEST_EPS)


def test_tightening_shrinks_the_boxes():
    _, on = _fista(ft.Verifier(), K=3)
    _, off = _fista(ft.Verifier(tighten_residuals=False), K=3)

    def width(iterates):
        return sum(np.sum(s.ub - s.lb) for s in iterates[1:])

    assert width(on) < width(off)


def test_only_successive_infinity_norm_differences_are_chained():
    vp = ft.Verifier()
    _, iterates = _fista(vp, K=2, solve=False)
    s1, s2 = iterates[1], iterates[2]
    for objective in (ft.norm(2 * (s2 - s1), np.inf), ft.norm(s2 - s1, 1), ft.norm([s2 - s1], 1)):
        vp.maximize(objective)
        vp.solve()
        assert not vp._residual_chain
    vp.maximize(ft.norm(s2 - s1, np.inf))
    vp.solve()
    assert s2.counter in vp._residual_chain


def test_other_objectives_clear_the_pending_residual():
    vp = ft.Verifier()
    _, iterates = _fista(vp, K=1, solve=False)
    vp.maximize(ft.norm(iterates[1] - iterates[0], np.inf))
    vp.maximize(0)
    vp.solve()
    assert not vp._residual_chain


def test_disabled_and_non_mip_backends_do_not_tighten():
    vp = ft.Verifier(tighten_residuals=False)
    _fista(vp, K=2)
    assert not vp._residual_chain
    sim = ft.Verifier(method="simulator")
    assert not sim.tighten_residuals


def test_a_looser_bound_does_not_overwrite_a_tighter_one():
    vp = ft.Verifier()
    _, iterates = _fista(vp, K=1, solve=False)
    s0, s1 = iterates
    vp.maximize(ft.norm(s1 - s0, np.inf))
    vp.solve()
    radius = vp._residual_chain[s1.counter][1]
    vp.maximize(ft.norm(s1 - s0, np.inf))
    vp._residual = (s1, s0)
    vp._chain_residual(10 * radius)
    assert vp._residual_chain[s1.counter][1] == radius
