"""Regression tests for parameter-dependent operations in simulator mode.

The examples are reduced from the optimal-control and network-optimization
tutorials. In both cases, a sampled parameter is not merely an input to an
algorithm step: it defines a saturation or projection bound that must be
evaluated separately for each sample.
"""

import numpy as np
import pytest

import fomtools as ft
from fomtools.proxable_functions import IndBox
from fomtools.verifier import Verifier


def test_optimal_control_stacked_box_without_metadata_uses_each_sample():
    """A symbolic box bound must be evaluated separately for every sample.

    This is a small analogue of ``docs/tutorials/optimal_control.py``, where
    the projection box is

        [x, u_min] <= w <= [x, u_max].

    The first coordinate is therefore fixed to the sampled value of ``x``.
    Samples alone must be enough to evaluate the box. Infinite default metadata
    must not produce NaNs during stack bound propagation.
    """
    x_samples = np.array([[-1.0], [2.0]])
    v_samples = np.array([[5.0, 5.0], [5.0, 5.0]])

    vp = Verifier(method="simulator", obbt_solver=None)

    x = vp.parameter(ndim=1, samples=x_samples)
    u_min = vp.parameter(ndim=1, lb=-0.5, ub=-0.5)
    u_max = vp.parameter(ndim=1, lb=1.5, ub=1.5)

    lower = vp.add_param_stack([x, u_min])
    upper = vp.add_param_stack([x, u_max])

    v = vp.parameter(ndim=2, samples=v_samples)
    projected = vp.prox(IndBox(l=lower, u=upper), v)

    expected_samples = np.array(
        [
            [-1.0, 1.5],
            [2.0, 1.5],
        ]
    )
    expected = vp.parameter(ndim=2, samples=expected_samples)
    vp.maximize(ft.norm(projected - expected, np.inf))

    residuals = vp.solve()

    assert np.isfinite(residuals).all()
    np.testing.assert_allclose(residuals, 0.0, atol=1e-7)


def test_optimal_control_stacked_box_ignores_global_sample_metadata():
    """Finite metadata must not replace per-sample optimal-control box limits.

    Adding the global range ``-1 <= x <= 2`` avoids the NaNs tested above, but
    the projection still needs the two actual values ``x=-1`` and ``x=2``.
    Clipping both rows to the common interval ``[-1, 2]`` is incorrect.
    """
    x_samples = np.array([[-1.0], [2.0]])
    v_samples = np.array([[5.0, 5.0], [5.0, 5.0]])

    vp = Verifier(method="simulator", obbt_solver=None)

    x = vp.parameter(ndim=1, lb=-1.0, ub=2.0, samples=x_samples)
    u_min = vp.parameter(ndim=1, lb=-0.5, ub=-0.5)
    u_max = vp.parameter(ndim=1, lb=1.5, ub=1.5)
    lower = vp.add_param_stack([x, u_min])
    upper = vp.add_param_stack([x, u_max])

    v = vp.parameter(ndim=2, samples=v_samples)
    projected = vp.prox(IndBox(l=lower, u=upper), v)

    expected_samples = np.array(
        [
            [-1.0, 1.5],
            [2.0, 1.5],
        ]
    )
    expected = vp.parameter(ndim=2, samples=expected_samples)
    vp.maximize(ft.norm(projected - expected, np.inf))

    residuals = vp.solve()

    np.testing.assert_allclose(residuals, 0.0, atol=1e-7)


def test_network_flow_projection_uses_sampled_capacities():
    """The network tutorial's flow capacity must be applied per sample.

    ``docs/tutorials/network_optimization.py`` constructs
    ``IndBox(l=0, u=c_param)``. If capacities are sampled in simulator mode,
    each row must be clipped to its own capacity vector rather than to the
    global upper metadata bound.
    """
    capacity_samples = np.array(
        [
            [0.5, 1.0],
            [2.0, 3.0],
        ]
    )
    flow_samples = np.full((2, 2), 5.0)

    vp = Verifier(method="simulator", obbt_solver=None)

    capacity = vp.parameter(
        ndim=2,
        lb=np.array([0.5, 1.0]),
        ub=np.array([2.0, 3.0]),
        samples=capacity_samples,
    )
    flow = vp.parameter(ndim=2, samples=flow_samples)
    projected = vp.prox(IndBox(l=0.0, u=capacity), flow)

    expected = vp.parameter(ndim=2, samples=capacity_samples)
    vp.maximize(ft.norm(projected - expected, np.inf))

    residuals = vp.solve()

    np.testing.assert_allclose(residuals, 0.0, atol=1e-7)


def test_parameter_dependent_satlin_evaluates_both_bounds_per_sample():
    """The primitive satlin underlying both tutorial projections is dynamic.

    Lower and upper bounds may both be sampled expressions. The simulator must
    evaluate those expressions for each row before applying ``clip``.
    """
    arg_samples = np.array(
        [
            [-5.0, 5.0],
            [-5.0, 5.0],
        ]
    )
    lower_samples = np.array(
        [
            [-2.0, 0.0],
            [1.0, 2.0],
        ]
    )
    upper_samples = np.array(
        [
            [-1.0, 1.0],
            [2.0, 3.0],
        ]
    )
    expected_samples = np.array(
        [
            [-2.0, 1.0],
            [1.0, 3.0],
        ]
    )

    vp = Verifier(method="simulator", obbt_solver=None)

    arg = vp.parameter(ndim=2, samples=arg_samples)
    lower = vp.parameter(
        ndim=2,
        lb=np.array([-2.0, 0.0]),
        ub=np.array([1.0, 2.0]),
        samples=lower_samples,
    )
    upper = vp.parameter(
        ndim=2,
        lb=np.array([-1.0, 1.0]),
        ub=np.array([2.0, 3.0]),
        samples=upper_samples,
    )
    projected = vp.satlin(arg, l=lower, u=upper)

    expected = vp.parameter(ndim=2, samples=expected_samples)
    vp.maximize(ft.norm(projected - expected, np.inf))

    residuals = vp.solve()

    np.testing.assert_allclose(residuals, 0.0, atol=1e-7)


def test_param_without_samples_raises_a_clear_error():
    """Bounds alone do not define a distribution: the simulator asks for samples."""
    v = Verifier(method="simulator", obbt_solver=None)
    x = v.parameter(ndim=3, lb=-1.0, ub=1.0)  # allowed: code generation never simulates
    z0 = v.iterate(ndim=3, init=np.zeros(3))
    v.implicit_linear(z0 + x, output_mat=np.eye(3))
    with pytest.raises(ValueError, match=r"1 parameter\(s\) \(ndim 3\) have no samples"):
        v.solve()


def test_param_with_wrong_sample_shape_raises():
    v = Verifier(method="simulator", obbt_solver=None)
    with pytest.raises(
        ValueError, match=r"samples must have shape \(num_samples, 3\), got \(10, 2\)"
    ):
        v.parameter(ndim=3, samples=np.zeros((10, 2)))


def test_param_pinned_by_equal_bounds_needs_no_samples():
    """A parameter fixed by equal finite bounds is a constant, so samples are optional."""
    v = Verifier(method="simulator", obbt_solver=None)
    x = v.parameter(ndim=1, samples=np.full((5, 1), 3.0))
    c = v.parameter(ndim=1, lb=2.0, ub=2.0)
    z0 = v.iterate(ndim=1, init=np.zeros(1))
    v.implicit_linear(z0 + x + c, output_mat=np.eye(1))
    residuals = np.asarray(v.solve())
    np.testing.assert_allclose(residuals[:, -1], 5.0)
